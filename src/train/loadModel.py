import torch
import torch.nn as nn

from src.models.modnet import MODNet

def makeStateDict(modelPath):
    modnet = MODNet(backbone_pretrained=False)

    torch.save(modnet.state_dict(), modelPath)

def loadStateDict(modelPath):
    modelState = torch.load(modelPath, map_location=torch.device('cpu'))

    state = {}

    prefix = "module."
    for key in modelState:
        stateKey = prefix + key
        state[stateKey] = modelState[key]
    return state

def saveState(model, optimizer, lr_scheduler, epoch, path):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
        }, 
        path
    )

def loadState(model, optimizer, lr_scheduler, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch = checkpoint['epoch']    
    return model, optimizer, lr_scheduler, epoch

def main():
    modelPath = "models/model.ckpt"
    pretrainedModelPath = "pretrained/modnet_webcam_portrait_matting.ckpt"

    makeStateDict(modelPath)

    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    state = loadStateDict(modelPath)
    stateKeys = list(state.keys())
    print(f"state keys {stateKeys[:5]}")

    modnet.load_state_dict(state)

    pretrainedState = torch.load(pretrainedModelPath, map_location=torch.device('cpu'))
    pretrainedStateKeys = list(pretrainedState.keys())
    print(f"pretrainedState keys {pretrainedStateKeys[:5]}")

    modnet.load_state_dict(pretrainedState)

    print(f"state {len(stateKeys)}, preptrainedState {len(pretrainedStateKeys)}, intersection {len(set(stateKeys) & set(pretrainedStateKeys))}")

if __name__ == "__main__":
    main()
