import torch
from torch.utils.data.dataloader import DataLoader
from model import Model
from train import eval 
from omegaconf import OmegaConf 
from dataset import ImageDataset 
import sys 

import argparse 

'''
    This script evaluates a given checkpoint model using a given configuration file
'''
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str) 
    parser.add_argument('--ckpt_path', type=str) 
    args = parser.parse_args() 

    conf_path = args.conf_path
    print(f'{conf_path=}')
    conf = OmegaConf.load(conf_path)
    # exit()
    val_dataset = ImageDataset(orig_img_path=conf.orig_datapath, sigma=conf.noise_sig, noised_img_path=conf.noised_datapath)

    val_loader = DataLoader(val_dataset, batch_size=1)

    ckpt_path = args.ckpt_path
    keys = torch.load(ckpt_path)

    model = Model(3, 3) 
    model.load_state_dict(keys)

    print(f'evaluation: {eval(val_loader, model, conf)}') 

if __name__ == '__main__': 
    main() 