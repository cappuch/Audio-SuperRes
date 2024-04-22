import utils.pre as pre
import os
import tqdm
import threading

formats = [
    'mp3',
    'wav',
    'flac',
    'm4a'
]

dir = 'test'

def process_audio(audio_file):
    pre.audio_to_spec(audio_file, f'specs/{os.path.basename(audio_file).split(".")[0]}.npy')

def process_batch(batch):
    threads = []
    for audio_file in batch:
        thread = threading.Thread(target=process_audio, args=(audio_file,))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()

audio_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.split('.')[-1] in formats]
batch_size = 4
num_batches = len(audio_files) // batch_size + (len(audio_files) % batch_size != 0)

for i in tqdm.tqdm(range(num_batches)):
    batch = audio_files[i * batch_size: (i + 1) * batch_size]
    process_batch(batch)
