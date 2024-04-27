from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os 
from pathlib import Path

from tqdm import tqdm, trange
class ImageDataset(Dataset): 
    def __init__(self, orig_img_path, sigma,patchnum=1,noised_img_path=None): 
        self.sigma = sigma / 255
        super(ImageDataset, self).__init__() 
        self.c_imgs = []
        self.n_imgs = [] 

        self.data_path = Path(orig_img_path)
        imDirec = os.listdir(orig_img_path)
        for p in range(patchnum):
            for i in tqdm(imDirec, desc='Patching Images'):
                oImg = cv2.imread(str(Path(orig_img_path, i)), 1)
                s = oImg.shape
                patch_x = np.random.randint(s[0]-128)
                patch_y = np.random.randint(s[1]-128) 
                patch=oImg[patch_x:(patch_x+128), patch_y:(patch_y+128)]
                self.c_imgs.append(patch)

                if noised_img_path is not None: 
                    noised_img = cv2.imread(str(Path(orig_img_path, i)), 1)
                    noised_patch = noised_img[patch_x:(patch_x+128), patch_y:(patch_y+128)] 
                    self.n_imgs.append(noised_patch)

                '''
                Add a 128x128 patch of each image to our data set 
                '''
        self.c_imgs =np.array(self.c_imgs)
        
        if noised_img_path is None: 
            self.n_imgs=np.copy(self.c_imgs)
            self._addNoise()
            self._augment_images()
        else: 
            self.n_imgs = np.array(self.n_imgs) 

        self.c_imgs=torch.from_numpy(np.moveaxis(self.c_imgs,3,1))
        self.n_imgs=torch.from_numpy(np.moveaxis(self.n_imgs,3,1))

        self.c_imgs = self.c_imgs / 255.
        self.n_imgs = self.n_imgs / 255.
        '''
        Creates a dictionary
        '''
        self.dicts=[]
        for i in range(len(self.c_imgs)):
            self.dicts.append({'noised_img': self.n_imgs[i], 'orig_img': self.c_imgs[i]})
    
    def __len__(self):
        return len(self.c_imgs)
    
    def _augment_images(self):
        '''
        Expected Input: n x c x h x w
        Goal: Randomly horizontally and vertically flip some of the images and
        add them to our data set 
        '''
        n=len(self.c_imgs)
        H_flip = np.random.binomial(1,.5,n)
        V_flip = np.random.binomial(1,.5,n)
        flip_ind = np.add(2*H_flip,3*V_flip)
        for i in tqdm(flip_ind, desc='Augmenting Images'): 
            if(flip_ind[i] == 2):
                self.c_imgs = np.append(self.c_imgs,[cv2.flip(self.c_imgs[i], 1)], axis=0)
                self.n_imgs = np.append(self.n_imgs,[cv2.flip(self.n_imgs[i], 1)], axis=0)
            elif(flip_ind[i] == 3):
                self.c_imgs = np.append(self.c_imgs,[cv2.flip(self.c_imgs[i], 0)], axis=0)
                self.n_imgs = np.append(self.n_imgs,[cv2.flip(self.n_imgs[i], 0)], axis=0)
            elif(flip_ind[i] == 5):
                self.c_imgs = np.append(self.c_imgs,[cv2.flip(self.c_imgs[i], -1)], axis=0)
                self.n_imgs = np.append(self.n_imgs,[cv2.flip(self.n_imgs[i], -1)], axis=0)

 
    
    def _addNoise(self):
        '''
        Implement Gaussian Noise based on Sigma Value 
        '''
        noise = np.random.normal(0,self.sigma,np.shape(self.c_imgs))
        self.n_imgs=np.add(self.c_imgs,noise)

    def __getitem__(self, idx):
        '''
            Returns a dictionary of format 
            {
                'noised_img': torch.Tensor(...), 
                'orig_img': torch.Tensor(...)
            }
        '''
        return self.dicts[idx]
    
