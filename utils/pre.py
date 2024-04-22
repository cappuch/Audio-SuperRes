import librosa
import numpy as np
import scipy.signal
import soundfile as sf

def audio_to_spec(audio, output_file=None):
    audio, sr = librosa.load(audio)
    S = librosa.stft(audio, n_fft=2048)
    magnitude_spec = np.abs(S)
    spec_db = librosa.power_to_db(magnitude_spec, ref=np.max)
    if output_file:
        np.save(output_file, spec_db)
    return spec_db

def spec_to_audio(spec, output_file):
    if type(spec) == str:
        spec_db = np.load(spec)
    else:
        spec_db = spec
    magnitude_spec = librosa.db_to_power(spec_db)
    S = magnitude_spec * np.exp(1j * np.angle(magnitude_spec))
    audio = librosa.istft(S, n_fft=2048)
    sf.write(output_file, audio, 22050)
