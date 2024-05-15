import torch
from torch.nn.functional import mse_loss

def forward(x, model, sig):
    '''
        Performs a forward pass of the input x through the model 

        Parameters: 
        - x: input to the model 
        - model: The score function model to be learned
        - sig: the annealed sigma for learning the model. 

        Returns: 
            (loss, u): 
            - loss: the AR-DAE loss between added noise and produced score 
            - u: the noise added to the image for training purpose
    '''
    u = torch.randn_like(x)

    x = x + sig * u

    score = model.forward(x)

    loss = mse_loss(-u, sig * score)
    return loss, u
