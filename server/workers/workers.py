import datetime
import re
# from lingua_franca.format import nice_time
# from lingua_franca.time import default_timezone
from text import text_to_sequence
import torch
import numpy as np
from scripts.inference_onnx import write_wav, load_onnx_tts, load_onnx_tts_unique



def worker_onnx(sentence, speaker_id, model, vocoder_model, use_aliases, new_speaker_ids, temperature, speaking_rate):

    global sessions

    def intersperse(lst, item):
        # Adds blank symbol
        result = [item] * (len(lst) * 2 + 1)
        result[1::2] = lst
        return result

    print(" > Model input: {}".format(sentence))
    print(" > Speaker Idx: {}".format(speaker_id))

    x = torch.tensor(
        intersperse(text_to_sequence(sentence, ["catalan_cleaners"]), 0),
        dtype=torch.long,
        device="cpu",
    )[None]

    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    x = x.detach().cpu().numpy()

    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device="cpu")
    x_lengths = np.array([x_lengths.item()], dtype=np.int64)

    if use_aliases:
        input_speaker_id = new_speaker_ids[speaker_id]
        print(input_speaker_id)
    else:
        input_speaker_id = speaker_id

    inputs = {
        "x": x,
        "x_lengths": x_lengths,
        "scales": np.array([temperature, speaking_rate], dtype=np.float32),
        "spks": np.repeat(input_speaker_id, x.shape[0]).astype(np.int64)
    }

    return write_wav(model, inputs=inputs, output_dir='', external_vocoder=vocoder_model)


def worker_onnx_audio(sentence, speaker_id, model_path, unique_model, vocoder_path, use_aliases, new_speaker_ids,
                      use_cuda, temperature, speaking_rate):

    def intersperse(lst, item):
        # Adds blank symbol
        result = [item] * (len(lst) * 2 + 1)
        result[1::2] = lst
        return result

    print(" > Model input: {}".format(sentence))
    print(" > Speaker Idx: {}".format(speaker_id))

    if unique_model:
        model = load_onnx_tts_unique(model_path=model_path, use_cuda=use_cuda)
        vocoder_model = None
    else:
        model, vocoder_model = load_onnx_tts(model_path=model_path, vocoder_path=vocoder_path, use_cuda=use_cuda)

    x = torch.tensor(
        intersperse(text_to_sequence(sentence, ["catalan_cleaners"]), 0),
        dtype=torch.long,
        device="cpu",
    )[None]

    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    x = x.detach().cpu().numpy()

    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device="cpu")
    x_lengths = np.array([x_lengths.item()], dtype=np.int64)

    if use_aliases:
        input_speaker_id = new_speaker_ids[speaker_id]
        print(input_speaker_id)
    else:
        input_speaker_id = speaker_id

    inputs = {
        "x": x,
        "x_lengths": x_lengths,
        "scales": np.array([temperature, speaking_rate], dtype=np.float32),
        "spks": np.repeat(input_speaker_id, x.shape[0]).astype(np.int64)
    }

    '''
    inputs = {
        "model1_x": x,
        "model1_x_lengths": x_lengths,
        "model1_scales": np.array([temperature, speaking_rate], dtype=np.float32),
        "model1_spks": np.repeat(input_speaker_id, x.shape[0]).astype(np.int64)
    }
    '''
    return write_wav(model, inputs=inputs, output_dir='', external_vocoder=vocoder_model)
