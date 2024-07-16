
# HPC Parallel Computing

This repository contains examples of parallel computing techniques for SLURM-based High-Performance Computing (HPC) servers, utilizing PyTorch Distributed Data Parallel (DDP) and Ray. These examples demonstrate how to efficiently distribute tasks and leverage the power of parallel computing to enhance computational performance.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [PyTorch DDP](#pytorch-ddp)
  - [Ray](#ray)
## Introduction

High-performance computing (HPC) allows for the processing of complex calculations at high speeds. This repository showcases examples of parallel computing on SLURM-based HPC servers using two popular libraries: PyTorch Distributed Data Parallel (DDP) and Ray. All code written is set up to function on the [Cannon Cluster](https://www.rc.fas.harvard.edu).


## Features

- **PyTorch DDP**: Demonstrates distributed training of deep learning models using PyTorch.
- **Ray**: Showcases parallel task execution and distributed computing using Ray for hyperparameter tuning of PyTorch models.

## Installation

To use the examples in this repository, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/nswood/HPC_Parallel_Computing.git
   cd HPC_Parallel_Computing
   ```
2. Create a new Conda environment from `environment.yml` and activate it:
    ```
    conda env create -f environment.yml
    conda activate hpc_env
    ```

## Usage

### PyTorch DDP
Navigate to the `DDP` directory and run `sbatch test.sh`. All modifications to the requested compute should be handled through the [`test.sh`](./DDP/test.sh) SLURM configuration file. Models are wrapped in [`Trainer.py`](./DDP/Trainer.py) to enable simple tracking of GPU allocations during parallelization. Extend [`Trainer.py`](./DDP/Trainer.py) to provide the necessary functionality for your model and save any needed metrics during training or evaluation.

### Ray
Navigate to the `Ray` directory and run `sbatch ray_slurm_template.sh`. All modifications to the requested compute should be handled through the [`ray_slurm_template.sh`](./Ray/ray_slurm_template.sh) SLURM configuration file. Ray is not natively compatible with SLURM-based servers, so manually start Ray instances on each compute node allocated through SLURM. Each Ray instance should be allocated all requested GPUs and CPUs. Allocations for individual trainings are handled through Ray, while total compute allocation is handled through SLURM.

To hyperparameter tune your own model, modify the [`ray_hyp_tune.py`](./Ray/ray_hyp_tune.py) file to incorporate your custom structure.


