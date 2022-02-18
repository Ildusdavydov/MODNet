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

class ImagesDataset(Dataset):
    def __init__(
        self,
        datasetDir: Union[str, Path],
        imageSize = 512
    ) -> None:

        self.datasetDir = Path(datasetDir)
        self.imageSize = imageSize

        self.image_names = glob.glob(f"{self.datasetDir}/*.png")

        self.augmentation = ImagesDataset.makeAugmentations(self.imageSize)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
        ])

    @staticmethod
    def makeAugmentations(imageSize):
        return albu.Compose([
                    albu.HorizontalFlip(),
                    albu.RandomGamma(gamma_limit=(50, 200)),
                    albu.OneOf([
                        albu.CLAHE(clip_limit=2),
                        albu.Sharpen(),
                        albu.Emboss(),
                        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                    ], p=0.3),
                    albu.HueSaturationValue(p=0.3),
                    albu.OneOf([
                        albu.MotionBlur(p=0.2),
                        albu.MedianBlur(blur_limit=3, p=0.1),
                        albu.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
                    # albu.OneOf([
                    #     albu.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.3),
                    #     albu.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.1),
                    #     albu.PiecewiseAffine(p=0.3),
                    # ], p=0.2),
                    albu.OneOf([
                        albu.Compose([
                            albu.SmallestMaxSize([imageSize, imageSize * 1.1, imageSize * 1.2]),
                            albu.RandomCrop(imageSize, imageSize)
                        ]),
                        albu.Compose([
                            albu.ShiftScaleRotate(scale_limit=0, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                            albu.Resize(imageSize, imageSize)
                        ])
                    ], p=1.0),
                    albu.GaussNoise(p=0.2),
                    albu.RandomRotate90()
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