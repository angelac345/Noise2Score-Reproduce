import torch
from model import Model
from train import eval 
from omegaconf import OmegaConf 
from dataset import ImageDataset 

conf = OmegaConf.load('conf.yaml')
val_dataset = ImageDataset(path=conf.eval_data_path, sigma=conf.noise_sig) 

keys = torch.load('/home/ac2323/4782/Noise2Score-Reproduce/ckpts/save-forward-scale1-train/save-forward-scale1-train-epoch=90-loss=962.2079467773438-psnr=-10175.068867653992.pth')

model = Model(3, 3) 
model.load_state_dict(keys)

print(f'evaluation: {eval(val_dataset, model, conf)}') 