import asyncio
import tempfile
import os
import io
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from starlette.websockets import WebSocket

from server.exceptions import SpeakerException
from server.helper.config import ConfigONNX
from server.workers.workers import worker_onnx_audio


async def play_audio(queue: asyncio.Queue, websocket: WebSocket):
    while True:
        # get the next audio chunk from the queue
        audio_chunk = await queue.get()

        # check if this is the end of the stream
        if audio_chunk is None:
            break

        # send the audio chunk to the client
        await websocket.send_bytes(audio_chunk)
        # print a message for debugging
        # print(f"Sent audio chunk of {len(audio_chunk)} bytes")
        # receive any data from the client (this will return None if the connection is closed)
        # TODO needs a timeout here in case the audio is not played (or finished?) within a given time
        data = await websocket.receive()
        # check if the connection is closed
        if data is None:
            break


async def generate_audio(sentences, speaker_id, audio_queue):
    config = ConfigONNX()
    model_tts = config.model_tts
    vocoder = config.vocoder
    speaking_rate = config.speech_speed
    temperature = config.temperature
    speaker_config_attributes = config.speakerConfigAttributes.__dict__

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        for sentence in sentences:
            sentence = sentence.strip()  # removes leading and trailing whitespaces
            if len(sentence) > 0:  # checks if sentence is not empty after removing whitespaces
                content = await loop.run_in_executor(
                    executor,
                    generate,
                    sentence,
                    speaker_config_attributes["speaker_ids"],
                    model_tts,
                    vocoder,
                    speaker_config_attributes["new_speaker_ids"],
                    speaker_config_attributes["use_aliases"],
                    speaker_id,
                    speaking_rate,
                    temperature
                )
                await audio_queue.put(content)

    await audio_queue.put(None)  # signal that we're done generating audio


def generate(sentence, speaker_ids, model_tts, vocoder, new_speaker_ids, use_aliases, speaker_id,
             speaking_rate, temperature):
    print(f"Processing sentence: {sentence}")

    if speaker_id not in speaker_ids.keys():
        raise SpeakerException(speaker_id=speaker_id)

    print(" > Model input: {}".format(sentence))
    print(" > Speaker Idx: {}".format(speaker_id))

    if use_aliases:
        input_speaker_id = new_speaker_ids[speaker_id]
    else:
        input_speaker_id = speaker_id

    # Create a temporary file name but do not open it
    temp_fd, tempfile_name = tempfile.mkstemp()
    os.close(temp_fd)

    p = Process(target=child_process, args=(tempfile_name, sentence, input_speaker_id, model_tts, vocoder,
                                            speaking_rate, temperature))
    p.start()
    p.join()

    # Read the data from the temp file
    with open(tempfile_name, 'rb') as tempf:
        out_data = tempf.read()

    # Remove the temporary file
    os.remove(tempfile_name)

    out = io.BytesIO(out_data)
    return out


def child_process(tempfile_name, sentence, input_speaker_id, model_tts, vocoder, speaking_rate, temperature):
    # sentence, speaker_id, model, vocoder_model, use_aliases, new_speaker_ids, temperature, speaking_rate
    wavs = worker_onnx_audio(sentence, speaker_id=input_speaker_id, model=model_tts, vocoder_model=vocoder,
                             temperature=temperature, speaking_rate=speaking_rate)
    with open(tempfile_name, 'wb') as tempf:
        model.save_wav(wavs, tempf)
