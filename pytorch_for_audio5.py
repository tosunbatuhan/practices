#-----------------------------------------------------------#
# Pre-processing Audio with Different Durations #
#-----------------------------------------------------------#

# some changes has done in previous file to do this

from pytorch_for_audio3 import UrbanSoundDataset
import torchaudio


if __name__ == "__main__":

    ANNOTATIONS_FILE = "..."
    AUDIO_DIR = "..."
    SAMPLE_RATE = 22055
    NUM_SAMPLES = 22055 # (1 sec)


    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=1024,hop_length=512,n_mels=64)

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES)

    print(f"There are {len(usd)} samples in the dataset.")

    signal, label = usd[0]


