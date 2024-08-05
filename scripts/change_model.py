import torch
import json

speakers = torch.load('../models/vits_ca/speakers.pth')
print(type(speakers))
conv = [line.strip().split(',') for line in open('speakers_conversion.csv').readlines()]
new_speakers = {}
for source, target in conv:
    id = speakers.get(source)
    if id:
        new_speakers[target] = source
with open('speaker_ids.json', 'w') as out:
    json.dump(new_speakers, out)
print(new_speakers)
