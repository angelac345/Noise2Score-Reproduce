import torch
from torch.nn.functional import mse_loss

def forward(x, model, sig):

    u = torch.randn_like(x)

    x = x + sig * u

    score = model.forward(x)

    loss = mse_loss(-u, sig * score)
    return loss, u
