
from algorithm import forward 
from dataset import ImageDataset
from model import Model
from tqdm import tqdm, trange
import torch 
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from pathlib import Path 
from argparse import ArgumentParser
import os 
import wandb
from omegaconf import OmegaConf

from util import psnr 

import time

def train(conf, train_dataset, eval_dataset, model, optimizer, scheduler): 

    if conf.use_wandb: 
        wandb.init(
            project="Noise2ScoreReproduce",
            name=conf.train_id,
            id=conf.train_id,
            resume='allow'
        )

    # setting up ckpt saving path
    save_path = Path('ckpts', conf.train_id)

    save_path.mkdir(parents=True, exist_ok=True)
    n_ckpts = len(os.listdir(save_path))
    
    sig_max, sig_min = conf.sig_max, conf.sig_min 

    print(f'starting training at: {n_ckpts} epoch')
    for i in trange(n_ckpts, conf.total_epochs, desc='Training Loop'): 

        q = (i+1) / len(train_dataset) 
        sig = sig_max * (1 - q) + sig_min * q

        
        loss_total = 0
        for data in tqdm(train_dataset, desc=f'epoch:{i}', position=-1): 

            optimizer.zero_grad()

            noised_img = data['noised_img'].float().unsqueeze(0)
            loss, u = forward(noised_img, model, sig=sig) 
            loss_total += loss 

            loss.backward()
            optimizer.step()

        scheduler.step() 

        log_dict = {'train_mean_loss': loss_total / len(train_dataset)}

        if i != 0 and i % conf.eval_freq == 0: 
            psnr_val = eval(eval_dataset, model, conf)
            log_dict['eval_psnr'] = psnr_val

            torch.save(model.state_dict(), Path(save_path, f'{conf.train_id}-epoch={i}-loss={loss_total}-psnr={psnr_val}.pth'))
        

        if conf.use_wandb: 
            wandb.log(log_dict)
        


def eval(eval_dataset, model, conf): 
    
    total_psnr = 0
    for data in tqdm(eval_dataset, desc='eval loop'): 

        noised, orig = data['noised_img'], data['orig_img'] 

        with torch.no_grad(): 
            out = model(noised.float().unsqueeze(0)) 
            recon = out.squeeze(0) * (conf.noise_sig**2) + noised 
        
        orig = orig * conf.peak
        recon = torch.clamp(recon * conf.peak, min=0, max=conf.peak) 
        total_psnr += psnr(orig, recon, peak=conf.peak)

    return total_psnr / len(eval_dataset)
        
def main(): 
    
    conf = OmegaConf.load('conf.yaml')
    train_dataset = ImageDataset(orig_img_path=conf.train_data_path, sigma=conf.noise_sig, patchnum=conf.patch_num) 
    val_dataset = ImageDataset(orig_img_path=conf.eval_data_path, sigma=conf.noise_sig, patchnum=conf.patch_num) 
    model = Model(in_channel=3, out_channel=3) 
    optimizer = Adam(model.parameters(), lr=conf.lr) 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    train(conf, train_dataset, val_dataset, model, optimizer, scheduler)


if __name__ == '__main__': 
    main()