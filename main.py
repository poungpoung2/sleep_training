from argparse import ArgumentParser
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from helper_function import seed_set
from modules import SleepDataModule , SleepModel
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint



class Audio_Config:
    sr = 44100
    n_fft = 1024
    hop_length = 500
    n_mels = 128
    fmin = 20
    fmax = 4000
    power = 2.0

def get_arguments():
    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)


    args = parser.parse_args()
    return args


def collect_audio_files(full_dir_path):
    sub_dirs = [d for d in full_dir_path.iterdir() if d.is_dir()]
    audio_files = []
    for dir in sub_dirs:
        files = list(dir.rglob('*.wav')) + list(dir.rglob('*.mp3'))
        audio_files.extend(files)

    return audio_files

def convet_audio_to_mel(full_dir_path, audio_config, mel_dir_path):
    audio_files = collect_audio_files(full_dir_path)

    columns = ["file_name", "label"]
    rows = []

    for path in audio_files:
        y, sr = librosa.load(path)

        name, idx = path.stem.split("_")

        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=audio_config.sr,
            n_fft=audio_config.n_fft,
            hop_length=audio_config.n_mels,
            fmin=audio_config.fmin,
            fmax=audio_config.fmax,
            power=audio_config.power,
        )

        mel_spec = mel_spec.astype("float32")
        mel_spec_db = librosa.power_to_db(mel_spec)
        if name in ("1", "0"):
            name = "snoring" if name == "1" else "non-snoring"

        mel_path = mel_dir_path / f"{name}_{idx}"
        np.save(mel_path, mel_spec_db)

        if name.lower() in ("snoring", "snore"):
            label = 1
        else:
            label = 0

        row = [mel_path.name, label]
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)

    return df

def tune_model(model, dm):
    trainer = Trainer(max_epochs=1)
    tuner = Tuner(trainer)

    lr_finder = tuner.lr_find(model, train_dataloaders=dm.train_dataloader())
    new_lr = lr_finder.suggestion()
    model.lr = new_lr if new_lr > 5e-5 else model.lr


def main():
    full_dataset_path = Path("../full_dataset_v1")
    mel_dir_path = Path("../Melspectogram_data")
    mel_dir_path.mkdir(exist_ok=True)
    checkpoint_dir_path = Path("checkpoints")
    checkpoint_dir_path.mkdir(exist_ok=True)

    seed_set()

    args = get_arguments()

    audio_config = Audio_Config()

    df = convet_audio_to_mel(full_dataset_path, audio_config, mel_dir_path)


    model = SleepModel(learning_rate=0.0022908676527677745, num_classes=2, global_epoch=0)    
    data_module = SleepDataModule(
        df=df, 
        split_prop=0.8,
        mel_dir_path=mel_dir_path,
        num_workers=args.num_workers
    )

    tune_model(model=model, dm=data_module)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir_path,
        filename='model_global_epoch{global_epoch:02d}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    trainer = Trainer(
        accelerator="cpu", 
        devices="auto", 
        default_root_dir=checkpoint_dir_path,
        profiler="simple",
        max_epochs=20,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()