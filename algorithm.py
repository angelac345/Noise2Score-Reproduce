import torch

def forward(x,model,sig):
    u=torch.randn_like(x)
    x=x+sig*u
    score=model.forward(x)
    lossfunc=torch.nn.MSELoss()
    loss = lossfunc(-forward(x,model,sig)[1],sig*forward(x,model,sig)[0])
    return loss
