import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
import os

from Trainer import trainer

# Simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gpu_id = 'cuda'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Generate random data
def generate_data(num_samples, input_size, num_classes):
    X = np.random.rand(num_samples, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples).astype(np.int64)
    y = torch.nn.functional.one_hot(torch.tensor(y), num_classes=num_classes).float()
    return torch.tensor(X), y

import argparse

p = argparse.ArgumentParser(description="DDP parser")

#DDP Congigs
p.add_argument('--gpu', default=None, type=int)
p.add_argument('--device', default='cuda', help='device')

p.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
p.add_argument('--rank', default=-1, type=int, 
                    help='node rank for distributed training')
p.add_argument('--dist-url', default='env://', type=str, 
                    help='url used to set up distributed training')
p.add_argument('--dist-backend', default='nccl', type=str, 
                    help='distributed backend')
p.add_argument('--local_rank', default=-1, type=int, 
                    help='local rank for distributed training')
p.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')

args = p.parse_args()
    
def ddp_setup():
    if 'SLURM_PROCID' in os.environ: # for slurm scheduler
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        print("Available GPUs:", torch.cuda.device_count())
        print("LOCAL_RANK:", args.gpu)
    else:
        print("Available GPUs:", torch.cuda.device_count())
        print("LOCAL_RANK:", int(os.environ["LOCAL_RANK"]))
    os.environ['RANK'] = str(args.rank)
    init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(args.gpu)
    

# Main function
def main():
    # DDP device params
    ddp_setup()
    # Hyperparameters
    input_size = 10
    hidden_size = 5
    output_size = 3
    num_samples = 1000
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Generate random data
    X, y = generate_data(num_samples, input_size, output_size)
    
    # Create DataLoader
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    
    X, y = generate_data(num_samples, input_size, output_size)
    test_dataset = TensorDataset(X, y)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    
    # Initialize model, loss function, and optimizer
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    
    
    # Train the model
    wrapped_model = trainer(model, train_loader, test_loader, optimizer, 5, 'test_run', criterion, args)
    wrapped_model.train(10)

if __name__ == "__main__":
    main()