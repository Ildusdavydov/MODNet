import argparse
from errno import EHOSTDOWN
import cv2
import numpy as np
from PIL import Image
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import albumentations as albu
from ffpyplayer.player import MediaPlayer

from src.models.modnet import MODNet

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelPath', type=str, required=True, help='path to trained model')
    parser.add_argument('--videoPath', type=str, required=True, help='path to video')
    args = parser.parse_args()
    return args

def getVideoSource(source, width, height):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def makePreprocessing(width, heigth, size):
    imageSize = max(width, heigth)
    preprocess = albu.Compose([
        albu.PadIfNeeded(imageSize, imageSize, border_mode=cv2.BORDER_CONSTANT, value = 0, mask_value=0),
        albu.Resize(size, size)
    ])
    return preprocess

def main():
    args = parseArgs()

    torch_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
        ]
    )

    print('Load pre-trained MODNet...')
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
        print('Use GPU...')
        modnet = modnet.cuda()
        modnet.load_state_dict(torch.load(args.modelPath))
    else:
        print('Use CPU...')
        modnet.load_state_dict(torch.load(args.modelPath, map_location=torch.device('cpu')))

    modnet.eval()

    print('Init WebCam...')
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    _, webFrame = cap.read()

    videoPlayer = getVideoSource(args.videoPath, webFrame.shape[1], webFrame.shape[0])
    audioPlayer = MediaPlayer(args.videoPath)

    print('Start matting...')
    while(True):
        videoRet, videoFrame = videoPlayer.read()
        if (videoRet == 0):
            videoPlayer = getVideoSource(args.videoPath, webFrame.shape[1], webFrame.shape[0])
            audioPlayer = MediaPlayer(args.videoPath)
            videoRet, videoFrame = videoPlayer.read()
        audio_frame, audioVal = audioPlayer.get_frame()

        _, webFrame = cap.read()
        webFrame = cv2.cvtColor(webFrame, cv2.COLOR_BGR2RGB)
        webFrame = cv2.flip(webFrame, 1)
        height = 256
        width = int(webFrame.shape[1] / webFrame.shape[0] * height)
        scaledWebFrame = cv2.resize(webFrame, (width, height), cv2.INTER_AREA)
        cropWidth = (width // 32) * 32
        left = (width - cropWidth) // 2
        right = left + cropWidth
        scaledWebFrame = scaledWebFrame[:, left:right, :]

        # frame_PIL = Image.fromarray(webFrame)
        frame_tensor = torch_transforms(scaledWebFrame)
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
        height = webFrame.shape[0]
        matteWidth = int(matte_np.shape[1] / matte_np.shape[0] * height)
        scaledMatte = cv2.resize(matte_np, (matteWidth, height), cv2.INTER_AREA)
        left = (webFrame.shape[1] - matteWidth) // 2
        right = left + matteWidth
        cutWebFrame = webFrame[:, left:right, :]

        height = webFrame.shape[0]
        width = int(videoFrame.shape[1] / videoFrame.shape[0] * height)
        scaledVideoFrame = cv2.resize(videoFrame, (width, height), cv2.INTER_AREA)
        left = (scaledVideoFrame.shape[1] - matteWidth) // 2
        right = left + matteWidth
        cutVideoFrame = scaledVideoFrame[:, left:right, :]
        print(f"scaledMatte {scaledMatte.shape}, cutWebFrame {cutWebFrame.shape}, cutVideoFrame {cutVideoFrame.shape}")
        fg_np = scaledMatte * cutWebFrame + (1 - scaledMatte) * cutVideoFrame
        fg_np = fg_np.astype(np.uint8)
        print(f"fg_np {fg_np.shape}, {fg_np.dtype}, min {fg_np.min()} max {fg_np.max()}")
        fg_np = cv2.cvtColor(fg_np, cv2.COLOR_RGB2BGR)

        if audioVal != 'eof' and audio_frame is not None:
            frame, t = audio_frame
            print("Frame:" + str(frame) + " T: " + str(t))

        cv2.imshow('MODNet - WebCam [Press \'Q\' To Exit]', fg_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoPlayer.release()
    cv2.destroyAllWindows()

    print('Exit...')

if __name__ == "__main__":
    main()