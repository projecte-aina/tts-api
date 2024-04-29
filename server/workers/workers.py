import datetime
import re
from lingua_franca.format import nice_time
from lingua_franca.time import default_timezone

def worker(sentence, speaker_id, model, use_aliases, new_speaker_ids):
    def substitute_time(sentence):
        # Regular expression to find time pattern (HH:MM)
        time_pattern = re.compile(r'((?<=\s)\d{1,2}):(\d{2}(?=\s))')

        # Find all matches of time pattern in the sentence
        matches = re.findall(time_pattern, sentence)

        if not matches:
            return sentence

        sentence = re.sub(r'les\s+', '', sentence, count=1)

        # Iterate through matches and substitute with formatted time string
        for match in matches:
            H = int(match[0])
            M = int(match[1])
            dt = datetime.datetime(2017, 1, 31, H, M, 0, tzinfo=default_timezone())  # Using UTC timezone for simplicity
            formatted_time = nice_time(dt, lang="ca", use_24hour=True)  # Assuming you have a function to format time in Catalan
            sentence = sentence.replace(f'{match[0]}:{match[1]}', formatted_time)

        return sentence

    sentence = substitute_time(sentence)

    print(" > Model input: {}".format(sentence))
    print(" > Speaker Idx: {}".format(speaker_id))

    if use_aliases:
        input_speaker_id = new_speaker_ids[speaker_id]
    else:
        input_speaker_id = speaker_id

    wavs = model.tts(sentence, input_speaker_id)

    return wavs
