
from algorithm import forward 
from tqdm import tqdm, trange

from argparse import ArgumentParser

def train(total_epochs, dataset, model, optimizer, sig_max, sig_min): 

    for i in trange(total_epochs): 
        q = i / len(dataset) 
        sig = sig_max * (1 - q) + sig_min * q

        optimizer.zero_grad() 

        loss_total = 0
        for data in dataset: 


            loss = forward(data, model, sig=sig) 
            loss_total += loss 
        
        loss_mean = loss_total / len(dataset)
        loss_mean.backward()
        optimizer.step()


