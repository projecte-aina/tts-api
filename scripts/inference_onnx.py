from pathlib import Path
from time import perf_counter

import onnxruntime as ort
import torch

from io import BytesIO
from typing import List

import numpy as np
import scipy

# from matcha.cli import process_text


def validate_args(args):
    assert (
            args.text or args.file
    ), "Either text or file must be provided Matcha-T(ea)TTS need sometext to whisk the waveforms."
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
    assert args.speaking_rate >= 0, "Speaking rate must be greater than 0"
    if args.vocoder_path:
        voc_name = args.vocoder_name.lower()
        assert (
                    voc_name == 'vocos' or voc_name == 'hifigan'), "If you use an external vocoder, please, specify which one"
    return args


def vocos_inference(mel, model_vocoder, denoise):
    # sample_rate = 22050
    n_fft = 1024
    hop_length = 256
    win_length = n_fft
    '''
    input_info = model_vocoder.get_inputs()
    for input in input_info:
        print("Name:", input.name)
        print("Shape:", input.shape)
        print("Type:", input.type)
        print("-" * 20)
    '''
    # ONNX inference
    mag, x, y = model_vocoder.run(
        None,
        {
            "mels": mel  # mel['mels']
        },
    )

    # complex spectrogram from vocos output
    spectrogram = mag * (x + 1j * y)
    window = torch.hann_window(win_length)

    if denoise:
        # Vocoder bias
        mel_rand = torch.zeros_like(torch.tensor(mel))
        mag_bias, x_bias, y_bias = model_vocoder.run(
            None,
            {
                "mels": mel_rand.float().numpy()
            },
        )

        # complex spectrogram from vocos output
        spectrogram_bias = mag_bias * (x_bias + 1j * y_bias)

        # Denoising
        spec = torch.view_as_real(torch.tensor(spectrogram))
        # get magnitude of vocos spectrogram
        mag_spec = torch.sqrt(spec.pow(2).sum(-1))

        # get magnitude of bias spectrogram
        spec_bias = torch.view_as_real(torch.tensor(spectrogram_bias))
        mag_spec_bias = torch.sqrt(spec_bias.pow(2).sum(-1))

        # substract
        strength = 0.0025
        mag_spec_denoised = mag_spec - mag_spec_bias * strength
        mag_spec_denoised = torch.clamp(mag_spec_denoised, 0.0)

        # return to complex spectrogram from magnitude
        angle = torch.atan2(spec[..., -1], spec[..., 0])
        spectrogram = torch.complex(mag_spec_denoised * torch.cos(angle), mag_spec_denoised * torch.sin(angle))

    # Inverse stft
    pad = (win_length - hop_length) // 2
    spectrogram = torch.tensor(spectrogram)
    B, N, T = spectrogram.shape

    print("Spectrogram synthesized shape", spectrogram.shape)
    # Inverse FFT
    ifft = torch.fft.irfft(spectrogram, n_fft, dim=1, norm="backward")
    ifft = ifft * window[None, :, None]

    # Overlap and Add
    output_size = (T - 1) * hop_length + win_length
    y = torch.nn.functional.fold(
        ifft, output_size=(1, output_size), kernel_size=(1, win_length), stride=(1, hop_length),
    )[:, 0, 0, pad:-pad]

    # Window envelope
    window_sq = window.square().expand(1, T, -1).transpose(1, 2)
    window_envelope = torch.nn.functional.fold(
        window_sq, output_size=(1, output_size), kernel_size=(1, win_length), stride=(1, hop_length),
    ).squeeze()[pad:-pad]

    # Normalize
    assert (window_envelope > 1e-11).all()
    y = y / window_envelope

    return y


def write_wav(model, inputs, output_dir, external_vocoder=None):

    print("[🍵] Generating mel using Matcha")
    '''
    input_info =model.get_inputs()
    for input in input_info:
        print("Name:", input.name)
        print("Shape:", input.shape)
        print("Type:", input.type)
        print("-" * 20)
    '''

    if external_vocoder is not None:

        mel_t0 = perf_counter()
        mel, mel_length = model.run(None, inputs)
        mel_infer_secs = perf_counter() - mel_t0
        print("Generating waveform from mel using external vocoder")

        vocoder_t0 = perf_counter()
        wav = vocos_inference(mel, external_vocoder, denoise=True)
        vocoder_infer_secs = perf_counter() - vocoder_t0

        wav_length = mel_length * 256
        infer_secs = mel_infer_secs + vocoder_infer_secs

        print("wav length tensor shape")
        print(wav_length.shape)

        wav_secs = wav_length.sum() / 22050
        print(f"Inference seconds: {infer_secs}")
        print(f"Generated wav seconds: {wav_secs}")
        rtf = infer_secs / wav_secs
        if mel_infer_secs is not None:
            mel_rtf = mel_infer_secs / wav_secs
            print(f"Matcha RTF: {mel_rtf}")
        if vocoder_infer_secs is not None:
            vocoder_rtf = vocoder_infer_secs / wav_secs
            print(f"Vocoder RTF: {vocoder_rtf}")
        print(f"Overall RTF: {rtf}")

        wav = wav.squeeze(1)

    else:
        print("I entered the inference function!!")
        wav_t0 = perf_counter()
        out_model = model.run(None, inputs)  # the tensor array with audio values
        model_infer_secs = perf_counter() - wav_t0

        print("Inference time in seconds: ")
        print(model_infer_secs)

        num_spec_frames, wav = out_model
        wav_length = num_spec_frames * 256
        wav_secs = wav_length.sum() / 22050

        model_rtf = model_infer_secs / wav_secs
        print(f"Overall RTF: {model_rtf}")

    wav = wav.squeeze()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return wav[:wav_length[0]]


def write_mels(model, inputs, output_dir):
    t0 = perf_counter()
    mels, mel_lengths = model.run(None, inputs)
    infer_secs = perf_counter() - t0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, mel in enumerate(mels):
        output_stem = output_dir.joinpath(f"output_{i + 1}")
        # plot_spectrogram_to_numpy(mel.squeeze(), output_stem.with_suffix(".png"))
        np.save(output_stem.with_suffix(".numpy"), mel)

    wav_secs = (mel_lengths * 256).sum() / 22050
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    print(f"RTF: {rtf}")


# taken from: https://github.com/coqui-ai/TTS/tree/dev
def save_wav_scipy(*, wav: np.ndarray, path: str, sample_rate: int = None, pipe_out=None, **kwargs) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
        pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
    """
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

    wav_norm = wav_norm.astype(np.int16)

    if pipe_out:
        wav_buffer = BytesIO()
        scipy.io.wavfile.write(wav_buffer, sample_rate, wav_norm)
        wav_buffer.seek(0)
        pipe_out.buffer.write(wav_buffer.read())
    scipy.io.wavfile.write(path, sample_rate, wav_norm)


# taken from: https://github.com/coqui-ai/TTS/tree/dev
def save_wav(wav: List[int], path: str, pipe_out=None) -> None:
    """Save the waveform as a file.

    Args:
        wav (List[int]): waveform as a list of values.
        path (str): output path to save the waveform.
        pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
    """
    # if tensor convert to numpy
    if torch.is_tensor(wav):
        wav = wav.cpu().numpy()
    if isinstance(wav, list):
        wav = np.array(wav)
    save_wav_scipy(wav=wav, path=path, sample_rate=22050, pipe_out=pipe_out)


def load_onnx_tts(model_path, vocoder_path, use_cuda):

    s_opts = ort.SessionOptions()
    # s_opts.intra_op_num_threads = 1
    # s_opts.inter_op_num_threads = 1

    if use_cuda:
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    model_tts = ort.InferenceSession(model_path, s_opts, providers=providers)
    vocoder_tts = ort.InferenceSession(vocoder_path, s_opts, providers=providers)

    return model_tts, vocoder_tts


def load_onnx_tts_unique(model_path, use_cuda):

    s_opts = ort.SessionOptions()
    # s_opts.intra_op_num_threads = 1  # amb 8 varios cops tira a 0.29 segons  / amb 1 varios cops sobre 0.60 segons
    # s_opts.inter_op_num_threads = 1
    # s_opts.intra_op_num_threads = 3  # total number of CPU's

    if use_cuda:
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    model_tts = ort.InferenceSession(model_path, s_opts, providers=providers)

    return model_tts
