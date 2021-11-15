from torch.utils import data
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from dataloader_segm import Dataloader_segm
from config import Config
from utils.utils import dice_loss
from unet3d import Unet3d
from utils.log import Log


if __name__ == '__main__':

    
    device = torch.device("cuda:0")
    
    try:
        os.mkdir(Config.tmp_save_dir)
    except:
        pass
    
    loader = Dataloader_segm(split='train',path_to_data=Config.data_path)
    trainloader= data.DataLoader(loader, batch_size=Config.train_batch_size, num_workers=Config.train_num_workers, shuffle=True,drop_last=True)

    loader = Dataloader_segm(split='test',path_to_data=Config.data_path)
    testLoader= data.DataLoader(loader, batch_size=Config.test_batch_size, num_workers=Config.test_num_workers, shuffle=False,drop_last=False)


    model=Unet3d()
    
    model=model.to(device)


    optimizer = optim.Adam(model.parameters(),lr=Config. init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    scheduler=optim.lr_scheduler.StepLR(optimizer, Config.step_size, gamma=Config.gamma, last_epoch=-1)

    log = Log()
    for epoch_num in range(Config.max_epochs):
        
        model.train()
        N=len(trainloader)
        for it, (batch,lbls) in enumerate(trainloader):
            
            if it%50==0:
                print(str(it) + '/' + str(N))
            
           
            batch=batch.to(device)
            lbls=lbls.to(device)
            
            res=model(batch)
            
            res=torch.softmax(res,1)
            loss = dice_loss(res,lbls)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()
        









    
    
    
    
    
    
    
    
    
    
    
    

