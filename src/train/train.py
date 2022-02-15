import os
import argparse
import logging
import logging.handlers
import numpy as np
import statistics

import math
import cv2
import scipy
from scipy.ndimage import grey_dilation, grey_erosion
import copy


from segmentation_models_pytorch.utils.functional import iou

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import neptune.new as neptune

from src.models.modnet import MODNet
from src.trainer import supervised_training_iter,soc_adaptation_iter
from src.train.AiSegmentationDataset import AiSegmentationDataset
from src.train.loadModel import loadState, saveState





logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetPath', type=str, required=True, help='path to dataset')
    parser.add_argument('--imageSize', type=int, default=512, help='image size for training')
    parser.add_argument('--trimapWidth', type=int, default=5, help='trimap width')
    parser.add_argument('--batchCount', type=int, default=16, help='batches count')
    parser.add_argument('--modelsPath', type=str, required=True, help='path to save trained MODNet models and states')
    parser.add_argument('--trainStatePath', type=str, help='path to saved train state')
    parser.add_argument('--train', action='store_true', help='make only train step')
    parser.add_argument('--tune', action='store_true', help='make only tune step')
    args = parser.parse_args()
    return args

def logNeptune(neptuneRun, type: str, semantic_loss, detail_loss, matte_loss=None, semantic_iou=None):
    neptuneRun[f"training/{type}/semantic_loss"].log(semantic_loss)
    neptuneRun[f"training/{type}/detail_loss"].log(detail_loss)
    if matte_loss is not None:
        neptuneRun[f"training/{type}/matte_loss"].log(matte_loss)
    if semantic_iou is not None:
        neptuneRun[f"training/{type}/semantic_iou"].log(semantic_iou)

def logNeptuneTest(neptuneRun, semantic_iou):
    neptuneRun[f"test/epoch/semantic_iou"].log(semantic_iou)

def evaluateModel(modnet, testDataloader):
    modnet.eval()
    modnetCpu = modnet.module.to("cpu")
    ious = []
    for idx, (image, _, gt_matte) in enumerate(testDataloader):
        _, _, pred_matte = modnetCpu(image, True)
        ious.append(iou(pred_matte, gt_matte).item())
    modnet = modnet.to(device)

    semantic_iou = statistics.mean(ious)
    return semantic_iou

def train(modnet, trainStatePath: str, datasetPath: str, imageSize: int, trimapWidth: int, batch_size: int, modelsPath: str, device: str):
    lr = 0.01       # learn rate
    epochs = 40     # total epochs
    startEpoch = 0

    optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1, last_epoch=startEpoch)

    if trainStatePath is not None:
        _, _, _, startEpoch, _ = loadState(modnet, optimizer, lr_scheduler, trainStatePath)
        startEpoch += 1
        print(f"train state from {trainStatePath} is restored")

    dataset = AiSegmentationDataset(datasetPath, imageSize, trimapWidth)

    TEST_PART = 0.1
    indices = list(range(len(dataset)))
    split = int(np.floor(TEST_PART * len(dataset)))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    trainDataloader = DataLoader(dataset, batch_size, sampler=train_sampler, drop_last=True)
    testDataloader = DataLoader(dataset, batch_size, sampler=test_sampler, drop_last=True)

    project = 'motionlearning/modnet'
    api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NmJhMzJiMC1iY2M0LTQ5NGUtOGI0OS0yYzA4ZmNiNTRiMmEifQ=='
    neptuneRun = neptune.init(project = project,
                            api_token = api_token,
                            source_files=[])

    for epoch in range(startEpoch, epochs):
        for idx, (image, trimap, gt_matte) in enumerate(trainDataloader):
            image = image.to(device)
            trimap = trimap.to(device)
            gt_matte = gt_matte.to(device)
            semantic_loss, detail_loss, matte_loss, semantic_iou = supervised_training_iter(modnet, optimizer, image, trimap, gt_matte, semantic_scale=1, detail_scale=10, matte_scale=1)
            if idx % 100 == 0:
                logger.info(f'idx: {idx}, semantic_loss: {semantic_loss:.5f}, detail_loss: {detail_loss:.5f}, matte_loss: {matte_loss:.5f}, semantic_iou: {semantic_iou:.5f}')
                logNeptune(neptuneRun, "batch", semantic_loss, detail_loss, matte_loss, semantic_iou)
        logNeptune(neptuneRun, "epoch", semantic_loss, detail_loss, matte_loss, semantic_iou)

        lr_scheduler.step()

        modelPath = os.path.join(modelsPath, f"model_epoch{epoch}.ckpt")
        statePath = os.path.join(modelsPath, f"state_epoch{epoch}.ckpt")
        saveState(modnet, optimizer, lr_scheduler, epoch, statePath)
        torch.save(modnet.state_dict(), modelPath)
        logger.info(f"model saved to {modelPath}")

        semantic_iou = evaluateModel(modnet, testDataloader)
        logger.info(f'Epoch: {epoch}, semantic_iou: {semantic_iou:.5f}')
        logNeptuneTest(neptuneRun, semantic_iou)

    torch.save(modnet.state_dict(), os.path.join(modelsPath, "model.ckpt"))

    neptuneRun.stop()

    
def tune(modnet, trainStatePath: str, datasetPath: str, imageSize: int, trimapWidth: int, batch_size: int, modelsPath: str, device: str): 
    # SOC model
    lr = 0.00001    # learn rate
    epochs = 10     # total epochs
    startEpoch = 0

    optimizer = torch.optim.Adam(modnet.parameters(), lr=lr, betas=(0.9, 0.99))

    if trainStatePath is not None:
        loadState(modnet, None, None, trainStatePath)
        print(f"train state from {trainStatePath} is restored")

    dataset = AiSegmentationDataset(datasetPath, imageSize, trimapWidth)

    TEST_PART = 0.1
    indices = list(range(len(dataset)))
    split = int(np.floor(TEST_PART * len(dataset)))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    # debugStartIndex = 7500
    # train_sampler = SubsetRandomSampler(list(range(debugStartIndex * batch_size, len(dataset))))
    test_sampler = SubsetRandomSampler(test_indices)

    # trainDataloader = DataLoader(dataset, batch_size, sampler=train_sampler, drop_last=True)
    trainDataloader = DataLoader(dataset, batch_size, drop_last=True)
    testDataloader = DataLoader(dataset, batch_size, sampler=test_sampler, drop_last=True)

    project = "motionlearning/modnet-soc"
    api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NmJhMzJiMC1iY2M0LTQ5NGUtOGI0OS0yYzA4ZmNiNTRiMmEifQ=='
    neptuneRun = neptune.init(project = project,
                            api_token = api_token,
                            source_files=[])

    

    for epoch in range(startEpoch, epochs):
        backup_modnet = copy.deepcopy(modnet)
        for idx, (image, trimap, gt_matte) in enumerate(trainDataloader):
            image = image.to(device)
            
            soc_semantic_loss, soc_detail_loss = soc_adaptation_iter(modnet, backup_modnet, optimizer, image, soc_semantic_scale = 1.0, soc_detail_scale = 1.0)
            if math.isnan(soc_semantic_loss) or math.isnan(soc_detail_loss):
                print(f"soc_semantic_loss {soc_semantic_loss}, soc_detail_loss {soc_detail_loss} for image {idx} {image.shape}")
            if idx % 100 == 0:
                logger.info(f'idx: {idx}, soc_semantic_loss: {soc_semantic_loss:.5f}, soc_detail_loss: {soc_detail_loss:.5f}')
                logNeptune(neptuneRun, "batch", soc_semantic_loss, soc_detail_loss, None, None)
        logNeptune(neptuneRun, "epoch", soc_semantic_loss, soc_detail_loss)
    

        modelPath = os.path.join(modelsPath, f"model_tuned_epoch{epoch}.ckpt")
        statePath = os.path.join(modelsPath, f"tune_state_epoch{epoch}.ckpt")
        saveState(modnet, optimizer, None, epoch, statePath)
        torch.save(modnet.state_dict(), modelPath)
        logger.info(f"model saved to {modelPath}")

        semantic_iou = evaluateModel(modnet, testDataloader)
        logger.info(f'Epoch: {epoch}, semantic_iou: {semantic_iou:.5f}')
        logNeptuneTest(neptuneRun, semantic_iou)

    torch.save(modnet.state_dict(), os.path.join(modelsPath, "model_tuned.ckpt"))

def twoStepTrain(datasetPath: str, imageSize: int, trimapWidth: int, batch_size: int, trainStatePath: str, modelsPath: str):
    device = "cuda"

    modnet = MODNet(backbone_pretrained=True)
    modnet = nn.DataParallel(modnet).to(device)

    
    train(modnet, trainStatePath, datasetPath, imageSize, trimapWidth, batch_size, modelsPath, device) # 1st model
    
    tune(modnet, trainStatePath, datasetPath, imageSize, trimapWidth, batch_size, modelsPath, device)  # 2nd model (no true labels needed)

args = parseArgs()

if args.train:
    device = "cuda"
    modnet = MODNet(backbone_pretrained=True)
    modnet = nn.DataParallel(modnet).to(device)    
    train(modnet, args.trainStatePath, args.datasetPath, args.imageSize, args.trimapWidth, args.batchCount, args.modelsPath, device)
elif args.tune:
    device = "cuda"
    modnet = MODNet(backbone_pretrained=True)
    modnet = nn.DataParallel(modnet).to(device)    
    tune(modnet, args.trainStatePath, args.datasetPath, args.imageSize, args.trimapWidth, args.batchCount, args.modelsPath, device)
else:
    twoStepTrain(args.datasetPath, args.imageSize, args.trimapWidth, args.batchCount, args.trainStatePath, args.modelsPath)