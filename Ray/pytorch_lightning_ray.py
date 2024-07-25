import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
torch.set_float32_matmul_precision('high')
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split
from torchvision import datasets, transforms
import torchvision
import torch.nn.functional as F
import scipy
import os
import tempfile
import sys

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune.search.hyperopt import HyperOptSearch
from ray.train.torch import TorchTrainer


from torch.optim.lr_scheduler import OneCycleLR
import lightning as pl
# from lightning.callbacks import ModelCheckpoint
# from lightning.loggers import TensorBoardLogger
# from ray_lightning import RayPlugin

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Ray")
    parser.add_argument('--cpu_per_trial', type=int, default=3, help='Number of CPUs per trial')
    parser.add_argument('--gpu_per_trial', type=int, default=1, help='Number of GPUs per trial')
    
    return parser.parse_args()



class Args:
    pass


def load_test_data():
    # Load fake data for running a quick smoke-test.
    trainset = torchvision.datasets.FakeData(
        128, (3, 32, 32), num_classes=10, transform=transforms.ToTensor()
    )
    testset = torchvision.datasets.FakeData(
        16, (3, 32, 32), num_classes=10, transform=transforms.ToTensor()
    )
    return trainset, testset


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


def load_model(args):
    model = Net(args.l1, args.l2)
    return model

class MyModel(pl.LightningModule):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.model, self.optimizer = self.load_train_objs()
        self.criterion = nn.CrossEntropyLoss()

    def load_train_objs(self):
        model = load_model(self.args)

        lr = self.args.lr
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

        return model, optimizer

    def forward(self, x_pf, jet_features, v):
        return self.model(x_pf, jet_features=jet_features, mask=None, training=self.training, v=v)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        output = torch.squeeze(output, dim=1).type(torch.double)
        loss = self.criterion(output, labels.type(torch.LongTensor))
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        output = torch.squeeze(output, dim=1).type(torch.double)
        loss = self.criterion(output, labels.type(torch.LongTensor))

        output = torch.nn.functional.softmax(output)
        predicted = torch.argmax(output.data, 1)
        correct = (predicted == torch.argmax(labels)).sum().item()
        total = labels.size(0)
        accuracy = correct / total

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer

class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(MyDataModule, self).__init__()
        self.args = args

    
    def load_data(self):

        test_abs = int(len(trainset) * 0.8)
        train_subset, val_subset = random_split(
            trainset, [test_abs, len(trainset) - test_abs])

        trainloader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=0 if config["smoke_test"] else 8,
        )
        valloader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=0 if config["smoke_test"] else 8,
        )
        return trainloader, valloader

    def setup(self, stage=None):
        self.train_loader, self.val_loader = self.load_data()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

def train_PM_trans(config):
    
#     print(config)
    
    temp_args = vars(config['args'])
    
    for key, value in config.items():
        if key != 'args':
            temp_args[key] = value
            
    args = Args()
    for key, value in temp_args.items():
        if key != 'args':
            setattr(args, key, value)
    
    
    
    model = MyModel(args)
    dm = MyDataModule(args)
    
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(find_unused_parameters=True),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
        max_epochs = 10
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)
    
  
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        path = os.path.join(temp_checkpoint_dir, "checkpoint.ckpt")
        trainer.save_checkpoint(path)
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        train.report({"loss": checkpoint_callback.best_model_score}, checkpoint=checkpoint)

def main(num_samples=10, max_num_epochs=10):
    args = parse_args()
    print(f"CPUs per trial: {args.cpu_per_trial}")
    print(f"GPUs per trial: {args.gpu_per_trial}")
    smoke_test = True
    # We add function args to config for simple pass through to train function
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "smoke_test": smoke_test,
        "args": args
    }

    scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=1, reduction_factor=2,metric="val_loss", mode="min")
    
    hyperopt_search = HyperOptSearch(metric="val_loss", mode="min")
    
    scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"CPU": args.cpu_per_trial, "GPU": args.gpu_per_trial})
    
    run_config = RunConfig(storage_path="/n/home11/nswood/HPC_Parallel_Computing/storage",
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
        ),
    )
    
    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_PM_trans,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config":config},
        tune_config=tune.TuneConfig(
            search_alg=hyperopt_search,
            scheduler=scheduler,
            num_samples=num_samples,
        )
    )

    results = tuner.fit()
    best_result = results.get_best_result("val_loss", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["val_loss"]))
    print("Best trial final validation accuracy: {}".format(best_result.metrics["val_accuracy"]))

if __name__ == "__main__":
    main(num_samples=100, max_num_epochs=10)
