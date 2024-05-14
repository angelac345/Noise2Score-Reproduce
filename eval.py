import torch
from torch.utils.data.dataloader import DataLoader
from model import Model
from train import eval 
from omegaconf import OmegaConf 
from dataset import ImageDataset 
import sys 

conf_path = sys.argv[1] 
print(f'{conf_path=}')
conf = OmegaConf.load(conf_path)
# exit()
val_dataset = ImageDataset(orig_img_path=conf.orig_datapath, sigma=conf.noise_sig, noised_img_path=conf.noised_datapath)

val_loader = DataLoader(val_dataset, batch_size=1)

keys = torch.load('/home/ac2323/4782/Noise2Score-Reproduce/ckpts/single-patch-no-shuffle-v2/single-patch-no-shuffle-v2-epoch=99-loss=830.3386840820312-psnr=23.19546127319336.pth')

model = Model(3, 3) 
model.load_state_dict(keys)

print(f'evaluation: {eval(val_loader, model, conf)}') 