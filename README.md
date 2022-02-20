# Motion Learning project - ODS Pet Project Hackaton

We made a fork from original [MODNet](https://github.com/ZHKKKe/MODNet) and implement full train loop for two stages (train and SOC adaptation). Also we tested training and SOC adaptation on a datasets different from datasets from origianl repo and get model for portrait matting. Demo with placing a people from camera above video is prepared.

# Installation

```
git clone https://github.com/PP22MotionLearning/MODNet.git

pip3 install numpy
pip3 install segmentation-models-pytorch
pip3 install neptune-client
pip3 install gdown

mkdir models
cd models
gdown --id 1mshivzbkzsb_e97Xf7Y3u-1LKp0hvMkh
cd ..
mkdir data
cd data
gdown --id 1fCuzD2twSxfSqmM_H6FrLuAR3Qe6K7VF
cd ..
```

#Train

Kaggle collab is here: https://www.kaggle.com/kapulkin/modnet-training

# Run
```
python3 -m demo.background_replacement.run --modelPath models/model_tuned_epoch1.ckpt --videoPath data/2001.A.Space.Odyssey.1968.2.avi
```