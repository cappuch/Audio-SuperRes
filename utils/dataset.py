import torch
import numpy as np
import scipy.signal
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, spec_dir, upscale_factor, fixed_length):
        self.spec_dir = spec_dir
        self.upscale_factor = upscale_factor
        self.fixed_length = fixed_length
        self.spec_files = [os.path.join(spec_dir, f) for f in os.listdir(spec_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.spec_files)

    def __getitem__(self, index):
        spec_file = self.spec_files[index]
        hr_spec = np.load(spec_file)
        spec = np.load(spec_file)
        if spec.shape[1] < self.fixed_length:
            hr_spec = np.pad(hr_spec, ((0, 0), (0, self.fixed_length - spec.shape[1])))
            spec = np.pad(spec, ((0, 0), (0, self.fixed_length - spec.shape[1])))
        elif spec.shape[1] > self.fixed_length:
            hr_spec = hr_spec[:, :self.fixed_length]
            spec = spec[:, :self.fixed_length]
        lr_spec = scipy.signal.decimate(spec, self.upscale_factor, axis=0)
        noise = np.random.normal(0, 0.5, lr_spec.shape)
        lr_spec += noise
        lr_spec = scipy.ndimage.gaussian_filter(lr_spec, sigma=np.random.uniform(0.5, 2))
        lr_spec = scipy.ndimage.zoom(lr_spec, zoom=(hr_spec.shape[0] / lr_spec.shape[0], 1))
        lr_spec, hr_spec = torch.from_numpy(lr_spec).unsqueeze(0).float(), torch.from_numpy(hr_spec).unsqueeze(0).float()
        return {'lr_spec': lr_spec, 'hr_spec': hr_spec}
