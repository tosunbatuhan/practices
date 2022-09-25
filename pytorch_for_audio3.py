#-----------------------------------------------------------#
# Custom Audio PyTorch Dataset with Torchaudio #
#-----------------------------------------------------------#

import os
from torch.utils.data import Dataset
import pandas as pd # download if not exists
import torchaudio
import torch

class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation,
                target_sample_rate, num_samples, device):

        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples=num_samples

    def __len__(self):
        return len(self.annotations)

    # len(usd)

    def __getitem__(self,index):

        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        

        signal = self._resample_if_necessary(signal,sr)
        
        signal = self._mix_down_if_necessary(signal)
        
        signal = self._right_pad_if_necessary(signal)

        signal = self._cut_if_necessary(signal)

        signal = self.transformation(signal)

        return signal, label

    # a_list[index] -> a_list.__getitem__(1)

    def _cut_if_necessary(self, signal):

        # signal -> Tensor -> (1, num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal


    def _right_pad_if_necessary(self, signal):

        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # we need to rigtht pad
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples) #(left_ped, right_pad)
            signal = torch.nn.functional.pad(signal, last_dim_padding)

        return signal

    def _resample_if_necessary(self, signal, sr):

        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)

        return signal

    def _mix_down_if_necessary(self, signal):

        if signal.shape[0] > 1 :
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        return signal


    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index,5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index,6]

if __name__ == "__main__":

    ANNOTATIONS_FILE = "..."
    AUDIO_DIR = "..."

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)

    print(f"There are {len(usd)} samples in the dataset.")

    signal, label = usd[0]

