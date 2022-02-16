import os
import glob
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union
import albumentations as albu

import torch
import torchvision
from torch.utils.data import Dataset

from src.train.AiSegmentationDataset import AiSegmentationDataset

class ImagesDataset(Dataset):
    def __init__(
        self,
        datasetDir: Union[str, Path],
        imageSize = 512
    ) -> None:

        self.datasetDir = Path(datasetDir)
        self.imageSize = imageSize

        self.image_names = glob.glob(f"{self.datasetDir}/*.png")

        self.augmentation = AiSegmentationDataset.makeAugmentations(self.imageSize)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
        ])

    def __getitem__(self, index: int) -> torch.Tensor:
        image_pth = self.image_names[index]

        image = np.array(Image.open(image_pth).convert("RGB"))

        aug = self.augmentation(image=image)
        image = aug["image"]

        # cv2.imshow("image", image)
        # cv2.waitKey(0)

        image = self.transform(image)
        
        return image

    def __len__(self):
        return len(self.image_names)