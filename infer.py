import torch
import numpy as np
import scipy.signal
import os
from utils.model import Model 
from utils.pre import audio_to_spec, spec_to_audio
import soundfile as sf
import matplotlib.pyplot as plt
import librosa

def infer(model, lr_spec):
    print('Inference...')
    model.eval()
    with torch.no_grad():
        lr_spec = torch.from_numpy(lr_spec).unsqueeze(0).unsqueeze(0).float()
        output = model(lr_spec)
        sr_spec = output.squeeze(0).squeeze(0).numpy()
    return sr_spec

print('Loading model...')
model = Model(upscale_factor=6)
model.load_state_dict(torch.load('model.pth'))

print('Processing audio...')

audio_file = 'test.mp3'
y, sr = librosa.load(audio_file, sr=None)

duration_to_trim = 5  # in seconds
samples_to_trim = int(duration_to_trim * sr)
trimmed_audio = y[:samples_to_trim]
noise = np.random.normal(0, 0.5, trimmed_audio.shape)
trimmed_audio += noise
trimmed_audio_file = 'trimmed_audio.wav'
sf.write(trimmed_audio_file, trimmed_audio, sr)

lr_spec = audio_to_spec('trimmed_audio.wav', 'temp.npy')
audio_spec = np.load('temp.npy')

sr_spec = infer(model, lr_spec)
plt.figure(figsize=(10, 4))
plt.imshow(sr_spec, aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Super-resolution Spectrogram')
plt.colorbar(label='Amplitude')
plt.tight_layout()
plt.show()
spec_to_audio(sr_spec, 'output_audio.wav')
