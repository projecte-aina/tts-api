import time
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import torch

# torch.set_num_threads(1)

path = "/home/mllopart/PycharmProjects/ONNX/venv/lib/python3.10/site-packages/TTS/.models.json"

model_manager = ModelManager(path)

model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/vits")

syn = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
)

text1 = "The field of space exploration has continually fascinated humanity, igniting the collective imagination and driving scientific and technological advancement. From the first successful launch of Sputnik 1 by the USSR in 1957, it became clear that space was a new frontier, ripe for exploration. Space exploration has offered us a unique vantage point to better understand our universe, revealing startling and wondrous phenomena like black holes, nebulae, and countless galaxies far beyond our own. It has also allowed us to study our home planet in ways that would have been impossible from the ground, enhancing our understanding of Earth's atmosphere, weather systems, and the impact of human activity on the global environment."

start_time = time.time()
outputs1 = syn.tts(text1)
end_time = time.time()
print(f"Time taken for inference 1: {end_time - start_time} seconds")
syn.save_wav(outputs1, "normal_1.wav")

text2 = "Additionally, space exploration has been integral to technological innovation, bringing forth advancements that have had broad implications on Earth. For instance, technology originally developed for space exploration has found its way into products ranging from GPS devices to medical equipment. The demands of space travel have spurred the development of new materials, energy sources, and computing technologies. Moreover, the highly collaborative nature of space exploration has brought nations together, fostering scientific cooperation across political divides."

start_time = time.time()
outputs2 = syn.tts(text2)
end_time = time.time()
print(f"Time taken for inference 2: {end_time - start_time} seconds")
syn.save_wav(outputs2, "normal_2.wav")

text3 = "In the years to come, the exploration of space will take on new dimensions as private companies become increasingly involved. Pioneers like SpaceX and Blue Origin are democratizing access to space, opening up opportunities for scientific research, commercial activities, and even space tourism. Furthermore, ambitious projects such as the Artemis program aim to return humans to the moon and eventually take us to Mars. These endeavors pose monumental challenges, but they also hold the promise of unprecedented scientific discoveries and technological breakthroughs. It's an exciting time for space exploration, and the next chapter promises to be even more transformative than the last."

start_time = time.time()
outputs3 = syn.tts(text3)
end_time = time.time()
print(f"Time taken for inference 3: {end_time - start_time} seconds")
syn.save_wav(outputs3, "normal_3.wav")
