
import numpy as np 

def augment_images(img: np.ndarray) -> np.ndarray: 
    '''
        Expected Input: n x c x h x w
    '''
    n=len(img)
    H_flip = np.where(np.random.binomial(1,.5,n)==1)[0]
    V_flip = np.where(np.random.binomial(1,.5,n)==1)[0]
    img[H_flip] = img[H_flip, :, ::-1, ::-1]
    img[V_flip] = img[V_flip, ::-1, :,::-1]
    return img 

def psnr(img, ref): 
    
    mse = np.mean(np.square(img - ref)) 
    return 10 * np.log10(1 / mse) 