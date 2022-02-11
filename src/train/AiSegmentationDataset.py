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

from src.train.trimap import makeTrimap

class AiSegmentationDataset(Dataset):
    """A custom Dataset(torch.utils.data) implement three functions: __init__, __len__, and __getitem__.
    Datasets are created from PTFDataModule.
    """

    def __init__(
        self,
        datasetDir: Union[str, Path],
        imageSize = 512,
        trimapSize = 5
    ) -> None:

        self.datasetDir = Path(datasetDir)
        self.imageSize = imageSize
        self.trimapSize = trimapSize

        self.image_names = glob.glob(f"{self.datasetDir}/clip_img/*/clip_*/*.jpg")
        self.mask_names = []
        clipPrefixLength = len("clip_")
        for imagePath in self.image_names:
            imageDir, imageName = os.path.split(imagePath)
            imagePartDir, clipDirName = os.path.split(imageDir)
            _, imagePartName = os.path.split(imagePartDir)
            clipNumber = clipDirName[clipPrefixLength:]

            imageNumber, _ = os.path.splitext(imageName)

            mathPath = os.path.join(self.datasetDir, "matting", imagePartName, f"matting_{clipNumber}", f"{imageNumber}.png")
            self.mask_names.append(mathPath)

        self.augmentation = AiSegmentationDataset.makeAugmentations(self.imageSize)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
        ])
        self.transform2 = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
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
                        albu.RandomBrightnessContrast(),
                    ], p=0.3),
                    albu.HueSaturationValue(p=0.3),
                    albu.OneOf([
                        albu.MotionBlur(p=0.2),
                        albu.MedianBlur(blur_limit=3, p=0.1),
                        albu.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
                    albu.OneOf([
                        albu.Compose([
                            albu.SmallestMaxSize(imageSize),
                            albu.RandomCrop(imageSize, imageSize)
                        ]),
                        albu.Compose([
                            albu.LongestMaxSize(imageSize),
                            albu.PadIfNeeded(imageSize, imageSize, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
                        ])
                    ], p=1.0),
                    albu.OneOf([
                        albu.OpticalDistortion(p=0.3),
                        albu.GridDistortion(p=.1),
                        albu.PiecewiseAffine(p=0.3),
                    ], p=0.2),
                    albu.GaussNoise(p=0.2),
                    albu.RandomRotate90()
                ])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_pth = self.image_names[index]
        mask_pth = self.mask_names[index]

        image = np.array(Image.open(image_pth).convert("RGB"))

        matting = np.array(Image.open(mask_pth).convert("RGBA"))[:, :, 3]

        aug = self.augmentation(image=image, mask=matting)
        image, matting = aug["image"], aug["mask"]

        mask = np.zeros_like(matting, dtype=np.float)
        threshold = 50
        mask[matting < threshold] = 0.0
        mask[matting >= threshold] = 1.0

        trimap = makeTrimap(mask, self.trimapSize)

        matting = matting / 255
        # cv2.imshow("image", image)
        # cv2.imshow("matting", matting)
        # cv2.imshow("mask", mask)
        # cv2.imshow("trimap", trimap)
        # cv2.waitKey(0)

        image = self.transform(image)

        trimap = torch.from_numpy(trimap).float()
        trimap = torch.unsqueeze(trimap, 0)

        matting = torch.from_numpy(matting)
        matting = torch.unsqueeze(matting, 0).float()
        
        return image, trimap, matting

    def __len__(self):
        return len(self.image_names)