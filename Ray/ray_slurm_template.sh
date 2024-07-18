#!/bin/bash

# '''
# =================================

# Slurm only handels allocating initial resources. All management is handeled by Ray.

# Make one "task" on each node with all the GPUs you want

# This will start a Ray instance on each node with all the GPUs allocated. 

# Ray handles internally the GPU work allocation

# =================================
# '''

#SBATCH -p gpu_test
#SBATCH -t 00:05:00 
#SBATCH --job-name=ray

### Number of total nodes requested 
#SBATCH --nodes=1

### Number of total nodes requested 
#SBATCH --ntasks=1

### Leave this 
#SBATCH --tasks-per-node=1

### GPU per node requested
#SBATCH --gres=gpu:2


### CPU per node requested. Make sure to leave 1 for Ray worker to use
#SBATCH --cpus-per-task=10

### Memory requested
#SBATCH --mem=200G
#SBATCH --chdir=/n/home11/nswood/HPC_Parallel_Computing/
#SBATCH --output=slurm_monitoring/%x-%j.out


### GPU per node requested (both should be the same as above)

export SLURM_GPUS_PER_TASK=2




nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-
address)


### UPDATE WITH YOUR CONFIGURATIONS

source ~/.bashrc

source /n/holystore01/LABS/iaifi_lab/Users/nswood/mambaforge/etc/profile.d/conda.sh

conda activate tune_env

if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[2]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"
redis_password=$(uuidgen)
echo "redis_password: "$redis_password

nodeManagerPort=6700
objectManagerPort=6701
rayClientServerPort=10001
redisShardPorts=6702
minWorkerPort=10002
maxWorkerPort=19999



# ray disable-usage-stats

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" \
        --port=$port \
        --node-manager-port=$nodeManagerPort \
        --object-manager-port=$objectManagerPort \
        --ray-client-server-port=$rayClientServerPort \
        --redis-shard-ports=$redisShardPorts \
        --min-worker-port=$minWorkerPort \
        --max-worker-port=$maxWorkerPort \
        --redis-password=$redis_password \
        --num-cpus "${SLURM_CPUS_PER_TASK}" \
        --num-gpus "${SLURM_GPUS_PER_TASK}" \
        --temp-dir="/n/home11/nswood/HPC_Parallel_Computing/log" \
        --block &

sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))


for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --redis-password=$redis_password \
        --num-cpus "${SLURM_CPUS_PER_TASK}" \
        --num-gpus "${SLURM_GPUS_PER_TASK}" \
        --temp-dir="/n/home11/nswood/HPC_Parallel_Computing/log" \
        --block &
    sleep 5
done




# python -u Ray/simpler-trainer.py "$SLURM_CPUS_PER_TASK"
python -u Ray/ray_hyp_tune.py --gpu_per_trial 1 --cpu_per_trial 2

