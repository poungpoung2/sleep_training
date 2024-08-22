from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from helper_function import seed_set, get_best_checkpoint
from modules import SleepDataModule, SleepModel
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import warnings
import time
import matplotlib.pyplot as plt
from cam import get_CAM_Image


warnings.filterwarnings("ignore")


class Audio_Config:
    sr = 44100
    n_fft = 1024
    hop_length = 500
    n_mels = 128
    fmin = 20
    fmax = 5000
    power = 2.0

def get_arguments():
    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--load_mel", action="store_true", help="Load saved Mel-spectrograms instead of generating them")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--load_checkpoint", action="store_true", help="Load the trained checkpoint")
    parser.add_argument("--grad_cam", action="store_true", help="Load the trained checkpoint")



    args = parser.parse_args()
    return args

def collect_audio_files(full_dir_path):
    sub_dirs = [d for d in full_dir_path.iterdir() if d.is_dir()]
    audio_files = []
    for dir in sub_dirs:
        files = list(dir.rglob('*.wav')) + list(dir.rglob('*.mp3'))
        audio_files.extend(files)
    return audio_files

def load_mel_data(mel_dir_path):
    mel_files = list(mel_dir_path.glob("*.npy"))

    columns = ["file_name", "label"]
    rows = []

    for mel_path in mel_files:
        name, idx = mel_path.stem.split("_")

        label = 1 if name.lower() in ("snoring", "snore") else 0
        rows.append([mel_path.name, label])

    df = pd.DataFrame(rows, columns=columns)
    return df


def convert_audio_to_mel(full_dir_path, audio_config, mel_dir_path):

    audio_files = collect_audio_files(full_dir_path)

    columns = ["file_name", "label"]
    rows = []

    for path in audio_files:
        try:
            y, sr = librosa.load(path, sr=audio_config.sr)

            name, idx = path.stem.split("_")

            if name in ("1", "0"):
                name = "snoring" if name == "1" else "non-snoring"

            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=audio_config.sr,
                n_fft=audio_config.n_fft,
                hop_length=audio_config.hop_length,
                fmin=audio_config.fmin,
                fmax=audio_config.fmax,
                power=audio_config.power,
            )

            # Convert the melspectogram image to RGB format
            mel_spec_db = librosa.power_to_db(mel_spec).astype("float32")
            mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            mel_spec_cmap = plt.cm.viridis(mel_spec_normalized)
            # Remove alpha channel
            mel_spec_rgb = (mel_spec_cmap[:, :, 3] * 255).astype(np.uint8)

            mel_path = mel_dir_path / f"{name}_{idx}.npy"
            np.save(mel_path, mel_spec_rgb)

            label = 1 if name.lower() in ("snoring", "snore") else 0
            rows.append([mel_path.name, label])

        except Exception as e:
            print(f"Error processing {path}: {e}")

    df = pd.DataFrame(rows, columns=columns)
    return df


def tune_model(model, dm):
    trainer = Trainer(max_epochs=1)
    tuner = Tuner(trainer)


    lr_finder = tuner.lr_find(model, datamodule=dm)
    new_lr = lr_finder.suggestion()
    model.lr = new_lr if new_lr > 5e-5 else model.lr

def main():
    full_dataset_path = Path("../../full_dataset_v1")
    mel_dir_path = Path("../Melspectogram_data")
    mel_dir_path.mkdir(exist_ok=True)

    cam_dir_path = Path("../GradCAM")
    cam_dir_path.mkdir(exist_ok=True)

    checkpoint_dir_path = Path("checkpoints")
    checkpoint_dir_path.mkdir(exist_ok=True)

    start_time = time.time()

    seed_set()

    args = get_arguments()

    audio_config = Audio_Config()

    if args.load_mel:
        print("Loading Mel-spectrogram files...")
        df = load_mel_data(mel_dir_path)
    else:
        print("Reading audio files...")
        df = convert_audio_to_mel(full_dataset_path, audio_config, mel_dir_path)

    end_time = time.time()
    print(f"Time Taken for task {end_time - start_time}")

    model = SleepModel(learning_rate=0.0022908676527677745, num_classes=2, global_epoch=0)
    data_module = SleepDataModule(
        df=df, 
        split_prop=0.8,
        mel_dir_path=mel_dir_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir_path,
        filename='model_global_epoch{global_epoch:.0f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp", 
        devices=2, 
        precision=args.precision,
        accumulate_grad_batches=args.accumulation,
        default_root_dir=checkpoint_dir_path,
        profiler="simple",
        max_epochs=20,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    if args.load_checkpoint:
        print(f"Loading best checkpoint...")
        checkpoint_path = get_best_checkpoint(checkpoint_dir=checkpoint_dir_path)
        if checkpoint_path is not None:
            model = SleepModel.load_from_checkpoint(checkpoint_path)
        else:
            print("No checkpoint found, starting training from scratch.")

    
    trainer.fit(model=model, datamodule=data_module)

    if(args.grad_cam):
        print(f"Generating Grad-CAM images...")
        get_CAM_Image(df, pl_model=model, n_samples=16, cam_img_dir=cam_dir_path)


if __name__ == "__main__":
    main()
