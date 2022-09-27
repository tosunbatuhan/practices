#-----------------------------------------------------------#
# Pre-processing Audio for Deep Learning on GPU #
#-----------------------------------------------------------#

# some changes has done in previous file to do this

from pytorch_for_audio3 import UrbanSoundDataset
import torchaudio
import torch


if __name__ == "__main__":

    ANNOTATIONS_FILE = "..."
    AUDIO_DIR = "..."
    SAMPLE_RATE = 22055
    NUM_SAMPLES = 22055 # (1 sec)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=1024,hop_length=512,n_mels=64)

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    print(f"There are {len(usd)} samples in the dataset.")

    signal, label = usd[0]


