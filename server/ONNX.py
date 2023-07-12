import numpy as np
import os
import json
import sys
import time
import torch

from pathlib import Path
from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.utils.manage import ModelManager

torch.set_num_threads(1)

config = VitsConfig()
config.load_json("/home/mllopart/PycharmProjects/ONNX/models/vits_ca/config.json")
vits = Vits.init_from_config(config)
vits.load_checkpoint(config,  "/home/mllopart/PycharmProjects/ONNX/models/vits_ca/model_file.pth")

vits.export_onnx()
vits.load_onnx("coqui_vits.onnx")

text1 = "The field of space exploration has continually fascinated humanity, igniting the collective imagination and driving scientific and technological advancement. From the first successful launch of Sputnik 1 by the USSR in 1957, it became clear that space was a new frontier, ripe for exploration. Space exploration has offered us a unique vantage point to better understand our universe, revealing startling and wondrous phenomena like black holes, nebulae, and countless galaxies far beyond our own. It has also allowed us to study our home planet in ways that would have been impossible from the ground, enhancing our understanding of Earth's atmosphere, weather systems, and the impact of human activity on the global environment."
text_inputs1 = np.asarray(
    vits.tokenizer.text_to_ids(text1, language="en"),
    dtype=np.int64,
)[None, :]

start = time.time()
audio1 = vits.inference_onnx(text_inputs1)
end = time.time()
print("Inference 1 Time Taken: ", end - start, " seconds")

print(audio1.shape)
save_wav(wav=audio1[0], path="ONNX_1.wav", sample_rate=config.audio.sample_rate)

text2 = "Additionally, space exploration has been integral to technological innovation, bringing forth advancements that have had broad implications on Earth. For instance, technology originally developed for space exploration has found its way into products ranging from GPS devices to medical equipment. The demands of space travel have spurred the development of new materials, energy sources, and computing technologies. Moreover, the highly collaborative nature of space exploration has brought nations together, fostering scientific cooperation across political divides."
text_inputs2 = np.asarray(
    vits.tokenizer.text_to_ids(text2, language="en"),
    dtype=np.int64,
)[None, :]

start = time.time()
audio2 = vits.inference_onnx(text_inputs2)
end = time.time()
print("Inference 2 Time Taken: ", end - start, " seconds")

print(audio2.shape)
save_wav(wav=audio2[0], path="ONNX_2.wav", sample_rate=config.audio.sample_rate)

text3 = "In the years to come, the exploration of space will take on new dimensions as private companies become increasingly involved. Pioneers like SpaceX and Blue Origin are democratizing access to space, opening up opportunities for scientific research, commercial activities, and even space tourism. Furthermore, ambitious projects such as the Artemis program aim to return humans to the moon and eventually take us to Mars. These endeavors pose monumental challenges, but they also hold the promise of unprecedented scientific discoveries and technological breakthroughs. It's an exciting time for space exploration, and the next chapter promises to be even more transformative than the last."
text_inputs3 = np.asarray(
    vits.tokenizer.text_to_ids(text3, language="en"),
    dtype=np.int64,
)[None, :]

start = time.time()
audio3 = vits.inference_onnx(text_inputs3)
end = time.time()
print("Inference 3 Time Taken: ", end - start, " seconds")

print(audio3.shape)
save_wav(wav=audio3[0], path="ONNX_3.wav", sample_rate=config.audio.sample_rate)
