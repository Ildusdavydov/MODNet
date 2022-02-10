import cv2
import albumentations as albu

from src.train.AiSegmentationDataset import AiSegmentationDataset

imagePath = "data/image101_MODNet.jpg"
image = cv2.imread(imagePath)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

imageSize = 256

augmentations = albu.Compose([
    albu.Emboss(p=1)
])

for i in range(5):
    augmentedImage = augmentations(image=image)["image"]

    cv2.imshow("image", image)
    cv2.imshow("auged", augmentedImage)
    cv2.waitKey(0)
