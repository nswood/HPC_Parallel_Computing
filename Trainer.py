import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim

# DDP Imports
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import os
import numpy as np
import h5py
import json
import numpy as np
import numpy.random as random
import glob




class trainer:
    def __init__(
        self,
        model,
        train_data,
        val_data,
        optimizer,
        save_every,
        outdir, 
        loss,
        args
        
    ):
        self.gpu_id = args.gpu
        self.global_rank = args.rank
        self.model = model.to(self.gpu_id)
        self.model.gpu_id = self.gpu_id
        self.model = DDP(self.model, device_ids=[self.gpu_id],find_unused_parameters=False)

        
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.outdir = outdir
        self.loss = loss
        self.loss_vals_training = []
        self.loss_vals_validation = []
        self.best_val_loss = 10e10
        


    def _run_epoch_val(self, epoch):
       
        with torch.no_grad():
            b_sz = len(next(iter(self.val_data))[0])
            loss_validation = []
            self.model.train(False)
            
            for istep, (x,y) in enumerate(self.val_data):
                
                   
                if istep != len(self.train_data) - 1:
                    lt, output = self._run_batch_val(istep, x,y)
                    loss_validation.append(lt)
              
            epoch_val_loss = np.mean(loss_validation)            
            self.loss_vals_validation.append(epoch_val_loss)
            
        
    def _run_batch_val(self, istep, x, y):
        x = x.to(self.gpu_id)
        y = y.to(self.gpu_id)
        self.model.eval()
        for param in self.model.parameters():
            param.grad = None

        x = x.to(self.gpu_id)
        output = self.model(x)
        l = self.loss(output, y)
        torch.cuda.empty_cache()
        return l.item(),output
        
    def _run_batch_train(self, istep, x,y):
        
        self.model.train(True)
        for param in self.model.parameters():
            param.grad = None
        
        x = x.to(self.gpu_id)
        y = y.to(self.gpu_id)

        output = self.model(x)
        self.optimizer.zero_grad()
        l = self.loss(output, y)
        l.backward()
        self.optimizer.step()  
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        return l.item(), output

    def _run_epoch_train(self, epoch):
        
        b_sz = len(next(iter(self.train_data))[0])
        
        loss_training = []
        self.model.train(True)
        train_data_length = len(self.val_data)

        for istep, (x,y) in enumerate(self.train_data):
            if istep != len(self.train_data) - 1:
                lt, output = self._run_batch_train(istep, x,y)
                loss_training.append(lt)
                
        epoch_train_loss = np.mean(loss_training)
        self.loss_vals_training.append(epoch_train_loss)
    
    def get_normalisation_weight(self,len_current_samples, len_of_longest_samples):
        return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)
    
    def _save_snapshot(self, epoch):
        torch.save(self.model.state_dict(),"{}/epoch_{}_{}_loss_{}_{}.pth".format(self.outdir,epoch,'test_model',round(self.loss_vals_training[-1],4),round(self.loss_vals_validation[-1],4)))
        if self.global_rank == 0:
            print(f" Training snapshot saved")
 
    def train(self, max_epochs: int):
        self.model.train(True)
        np.random.seed(max_epochs)
        random.seed(max_epochs)
        
        model_dir = self.outdir
        os.system("mkdir -p ./"+model_dir)

        
        for epoch in range(max_epochs):
            if epoch<max_epochs:

                self._run_epoch_train(epoch)
                self._run_epoch_val(epoch)
                if self.global_rank == 0:
                    print(f"[GPU{self.global_rank}] Epoch {epoch} | Steps: {len(self.train_data)} | Train Loss: {round(self.loss_vals_training[-1],4)} | Val Loss: {round(self.loss_vals_validation[-1],4)}")

                if self.global_rank == 0:

                    if (epoch % self.save_every == 0 or epoch == max_epochs-1):
                        self._save_snapshot(epoch)   

                    if self.loss_vals_validation[-1] < self.best_val_loss:
                        self.best_val_loss = self.loss_vals_validation[-1]
                        torch.save(self.model.state_dict(),"{}/best_model.pth".format(self.outdir))
                        
        torch.cuda.empty_cache()
