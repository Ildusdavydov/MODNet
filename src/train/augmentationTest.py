import cv2
import albumentations as albu

from src.train.AiSegmentationDataset import AiSegmentationDataset

imagePath = "data/image101_MODNet.jpg"
image = cv2.imread(imagePath)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

imageSize = 256

augmentations = albu.Compose([
    albu.GaussNoise(p=1)
])

# augmentations = albu.Compose([
#                     albu.HorizontalFlip(),
#                     albu.RandomGamma(gamma_limit=(50, 200)),
#                     albu.GaussNoise(p=0.2),
#                     albu.OneOf([
#                         albu.MotionBlur(p=0.2),
#                         albu.MedianBlur(blur_limit=3, p=0.1),
#                         albu.Blur(blur_limit=3, p=0.1),
#                     ], p=0.2),
#                     albu.OneOf([
#                         albu.Compose([
#                             albu.SmallestMaxSize(imageSize),
#                             albu.RandomCrop(imageSize, imageSize)
#                         ]),
#                         albu.Compose([
#                             albu.LongestMaxSize(imageSize),
#                             albu.PadIfNeeded(imageSize, imageSize, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
#                         ])
#                     ], p=1.0),
#                     albu.OneOf([
#                         albu.OpticalDistortion(p=0.3),
#                         albu.GridDistortion(p=.1),
#                         albu.PiecewiseAffine(p=0.3),
#                     ], p=0.2),
#                     albu.OneOf([
#                         albu.CLAHE(clip_limit=2),
#                         albu.Sharpen(),
#                         albu.Emboss(),
#                         albu.RandomBrightnessContrast(),
#                     ], p=0.3),
#                     albu.HueSaturationValue(p=0.3),
#                     albu.RandomRotate90()
#                 ])

for i in range(5):
    augmentedImage = augmentations(image=image)["image"]

    cv2.imshow("image", image)
    cv2.imshow("auged", augmentedImage)
    cv2.waitKey(0)
