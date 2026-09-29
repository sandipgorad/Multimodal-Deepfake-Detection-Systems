import os
import pandas as pd
import numpy as np
import cv2
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging


class CustomDataset(Dataset):
    def __init__(self, data_csv, file_dir, transform=None, file_type='image'):
        """
        Initializing CustomDataset.
        :param data_csv: Path to the CSV file containing file paths and labels
        :param file_dir: Directory where the files are stored
        :param transform: Transformations to apply to the files
        :param file_type: Type of file ('image', 'audio', or 'video')
        """
        self.logger = logging.getLogger('data_loader_logger')
        self.data_csv = data_csv
        self.file_dir = file_dir
        self.transform = transform
        self.file_type = file_type

        try:
            self.data = pd.read_csv(data_csv)
            self.logger.info(f"Loaded data from {data_csv}")
        except Exception as e:
            self.logger.error(
                f"Error loading data from {data_csv}: {e}",
                exc_info=True
            )
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            file_name = os.path.join(
                self.file_dir,
                self.data.iloc[idx, 0]
            )
            label = self.data.iloc[idx, 1]

            if self.file_type == 'image':
                file = cv2.imread(file_name)
                file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)

                if self.transform:
                    file = self.transform(file)

            elif self.file_type == 'audio':
                file, _ = torchaudio.load(file_name)
                file = torchaudio.transforms.Resample(
                    orig_freq=44100,
                    new_freq=16000
                )(file)

            elif self.file_type == 'video':
                cap = cv2.VideoCapture(file_name)
                frames = []

                while cap.isOpened():
                    ret, frame = cap.read()

                    if ret:
                        frame = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB
                        )
                        frames.append(frame)
                    else:
                        break

                cap.release()
                file = np.array(frames)

            self.logger.debug(
                f"Loaded {self.file_type} {file_name} "
                f"with label {label}"
            )

            return file, label

        except Exception as e:
            self.logger.error(
                f"Error loading {self.file_type} "
                f"{file_name}: {e}",
                exc_info=True
            )
            raise


def load_csv_data(file_path):
    return pd.read_csv(file_path)


def create_dataloader(
    data_csv,
    file_dir,
    file_type='image',
    batch_size=32,
    shuffle=True,
    num_workers=4
):
    """
    Creating DataLoader.
    """

    logger = logging.getLogger('data_loader_logger')

    logger.info(
        f"Creating DataLoader with data_csv={data_csv}, "
        f"file_dir={file_dir}, file_type={file_type}, "
        f"batch_size={batch_size}, shuffle={shuffle}, "
        f"num_workers={num_workers}"
    )

    try:
        if file_type == 'image':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            transform = None

        dataset = CustomDataset(
            data_csv=data_csv,
            file_dir=file_dir,
            transform=transform,
            file_type=file_type
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        logger.info("DataLoader created successfully")

        return dataloader

    except Exception as e:
        logger.error(
            f"Error creating DataLoader: {e}",
            exc_info=True
        )
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    image_csv = os.path.join(
        'data/processed',
        'processed_images.csv'
    )
    image_dir = os.path.join(
        'data/processed',
        'images'
    )

    image_dataloader = create_dataloader(
        image_csv,
        image_dir,
        file_type='image'
    )

    audio_csv = os.path.join(
        'data/processed',
        'processed_audios.csv'
    )
    audio_dir = os.path.join(
        'data/processed',
        'audios'
    )

    audio_dataloader = create_dataloader(
        audio_csv,
        audio_dir,
        file_type='audio'
    )

    video_csv = os.path.join(
        'data/processed',
        'processed_videos.csv'
    )
    video_dir = os.path.join(
        'data/processed',
        'videos'
    )

    video_dataloader = create_dataloader(
        video_csv,
        video_dir,
        file_type='video'
    )

    for images, labels in image_dataloader:
        print(f'Image batch shape: {images.size()}')
        print(f'Label batch shape: {labels.size()}')
        break

    for audios, labels in audio_dataloader:
        print(f'Audio batch shape: {audios.size()}')
        print(f'Label batch shape: {labels.size()}')
        break

    for videos, labels in video_dataloader:
        print(f'Video batch shape: {videos.shape}')
        print(f'Label batch shape: {labels.shape}')
        break