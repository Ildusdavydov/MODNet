from errno import EHOSTDOWN
import cv2
import numpy as np
from PIL import Image
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import albumentations as albu

from src.models.modnet import MODNet


torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

def makePreprocessing(width, heigth, size):
    imageSize = max(width, heigth)
    preprocess = albu.Compose([
        albu.PadIfNeeded(imageSize, imageSize, border_mode=cv2.BORDER_CONSTANT, value = 0, mask_value=0),
        albu.Resize(imageSize, imageSize)
    ])
    return preprocess

print('Load pre-trained MODNet...')
pretrained_ckpt = './pretrained/modnet_webcam_portrait_matting.ckpt'
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

GPU = True if torch.cuda.device_count() > 0 else False
if GPU:
    print('Use GPU...')
    modnet = modnet.cuda()
    modnet.load_state_dict(torch.load(pretrained_ckpt))
else:
    print('Use CPU...')
    modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))

modnet.eval()

print('Init WebCam...')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

preprocess = None

print('Start matting...')
while(True):
    _, frame_np = cap.read()
    if preprocess is None:
        print(f"image size {frame_np.shape[1]}x{frame_np.shape[0]}")
        preprocess = makePreprocessing(frame_np.shape[1], frame_np.shape[0], 256)
    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
    frame_np = frame_np[:, 120:792, :]
    print(f"image size {frame_np.shape[1]}x{frame_np.shape[0]}")
    frame_np = cv2.flip(frame_np, 1)

    # frame_np = preprocess(image=frame_np)["image"]

    frame_PIL = Image.fromarray(frame_np)
    frame_tensor = torch_transforms(frame_PIL)
    frame_tensor = frame_tensor[None, :, :, :]
    if GPU:
        frame_tensor = frame_tensor.cuda()
    
    with torch.no_grad():
        start = time.time()
        _, _, matte_tensor = modnet(frame_tensor, True)
        end = time.time()
        print(f"inference {end - start}")

    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
    fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
    view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))
    view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)

    cv2.imshow('MODNet - WebCam [Press \'Q\' To Exit]', view_np)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Exit...')
