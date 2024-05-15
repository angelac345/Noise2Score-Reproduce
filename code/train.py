
from algorithm import forward 
from dataset import ImageDataset
from model import Model
from tqdm import tqdm, trange
import torch 
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data.dataloader import DataLoader
import sys

import numpy as np 
import cv2 

from pathlib import Path 
from argparse import ArgumentParser
import os 
import wandb
from omegaconf import OmegaConf

from util import psnr 

import time

def train(conf, train_dataloader, eval_dataloader, model, optimizer, scheduler): 

    '''
        Parameters: 
        - conf: name of the config file that contains settings for this round of training 
        - train_dataloader: the dataloader that contains training data
        - eval_dataloader: the dataloader that contains validation data 
        - model: the model to train with this training loop 
        - optimizer: the optimizer used for training 
        - scheduler: learning rate scheduler used for training 
    
        
    '''
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

    conf_save = Path(save_path, 'conf.yaml') 
    if not conf_save.exists(): 
        OmegaConf.save(conf, Path(save_path, 'conf.yaml'))

    # check for existing checkpoints in the same training run, if there exists, 
    # load the last checkpoint and train from there 
    existing_ckpts = [p for p in os.listdir(save_path) if 'conf' not in p]
    n_ckpts = len(existing_ckpts)

    if n_ckpts > 0: 
        keys = torch.load(str(Path(save_path, existing_ckpts[-1]))) 
        model.load_state_dict(keys)

    # sigma annealing 
    sig_max, sig_min = conf.sig_max, conf.sig_min 

    print(f'starting training at: {n_ckpts} epoch') 
    for i in trange(n_ckpts, conf.total_epochs, desc='Training Loop', ncols=0, dynamic_ncols=False): 

        q = (i+1) / len(train_dataloader)
        sig = sig_max * (1 - q) + sig_min * q

        
        loss_total = 0
        for data in tqdm(train_dataloader, desc=f'epoch:{i}', position=-1, ncols=0, dynamic_ncols=False): 

            optimizer.zero_grad()

            noised_img = data['noised_img'].float()
            loss, u = forward(noised_img, model, sig=sig) 
            loss_total += loss 

            loss.backward()
            optimizer.step()

        scheduler.step() 

        log_dict = {'train_mean_loss': loss_total / len(train_dataloader)}

        # evaluating the current model based on the evaluation 
        if i != 0 and i % conf.eval_freq == 0: 
            psnr_val = eval(eval_dataloader, model, conf)
            log_dict['eval_psnr'] = psnr_val

            torch.save(model.state_dict(), Path(save_path, f'{conf.train_id}-epoch={i}-loss={loss_total}-psnr={psnr_val}.pth'))
        

        if conf.use_wandb: 
            wandb.log(log_dict)


def eval(eval_dataloader, model, conf): 
    '''
        Evaluates the model provided using `conf` config on `eval_dataloader` 

        - eval_dataloader: the dataloader containing validation data or evaluation data
        - model: the model to evaluate
        - conf: configurations containing the settings used for evaluation 

    '''
    orig_noised_psnr = 0
    orig_recon_psnr = 0
    recon_noised_psnr = 0
    for i, data in tqdm(enumerate(eval_dataloader), desc='eval loop'): 

        noised, orig = data['noised_img'], data['orig_img'] 


        with torch.no_grad(): 
            out = model(noised.float()) 
            recon = out.squeeze(0) * ((conf.noise_sig/255)**2) + noised 
        
        orig = orig * conf.peak
        recon = torch.clamp(recon * conf.peak, min=0, max=conf.peak) 
        noised = noised * conf.peak

        if conf.image_output: 
            visualize  = torch.cat([orig, recon, noised], dim=-1).squeeze(0).numpy()
            visualize = np.moveaxis(visualize, 0, 2)
            cv2.imwrite(f'/home/ac2323/4782/Noise2Score-Reproduce/img_output/img{i}.png', visualize)

        orig_recon_psnr += psnr(recon, orig, peak=conf.peak)
        orig_noised_psnr += psnr(orig, noised, peak=conf.peak)
        recon_noised_psnr += psnr(recon, noised, peak=conf.peak)

    print(f'{orig_noised_psnr/ len(eval_dataloader)=}, {orig_recon_psnr/ len(eval_dataloader)=}, {recon_noised_psnr/ len(eval_dataloader)=}')
    return orig_recon_psnr / len(eval_dataloader)
        
def main(): 
    conf_path = Path(Path(__file__).parent.parent, 'conf', sys.argv[1])
    
    conf = OmegaConf.load(conf_path)

    train_dataset = ImageDataset(
        orig_img_path=conf.train_data_path, 
        sigma=conf.noise_sig, 
        patchnum=conf.patch_num
    ) 
    val_dataset = ImageDataset(
        orig_img_path=conf.eval_orig_data_path, 
        sigma=conf.noise_sig, 
        patchnum=conf.patch_num, 
        noised_img_path=conf.eval_noised_data_path
    ) 

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=conf.train_batch_size, 
        shuffle=conf.shuffle
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=conf.val_batch_size, 
        shuffle=conf.shuffle
    )
    
    model = Model(in_channel=3, out_channel=3) 
    optimizer = Adam(model.parameters(), lr=conf.lr) 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    print(f'{conf.train_id=}')
    train(conf, train_dataloader, val_dataloader, model, optimizer, scheduler)


if __name__ == '__main__': 
    main()