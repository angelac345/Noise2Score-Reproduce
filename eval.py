import torch
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

keys = torch.load('/home/ac2323/4782/Noise2Score-Reproduce/ckpts/save-forward-Upsample-trainv2/save-forward-Upsample-trainv2-epoch=99-loss=958.1229858398438-psnr=9.700337075125885.pth')

model = Model(3, 3) 
model.load_state_dict(keys)

print(f'evaluation: {eval(val_dataset, model, conf)}') 