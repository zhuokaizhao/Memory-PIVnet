#!/bin/bash
#SBATCH --job-name=train    # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks across all nodes
#SBATCH --cpus-per-task=1            # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                     # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition general
#SBATCH --gres=gpu:rtx8000:1         # number of gpus per node
#SBATCH --time=04:00:00              # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin            # send email when job begins
#SBATCH --mail-type=end              # send email when job ends
#SBATCH --mail-user=zhuokai@cs.uchicago.edu

conda activate pt-env
srun python /net/scratch/zhuokai/Memory-PIVnet/main.py --mode train --network-model memory-piv-net --train-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/Isotropic_1024/PT_Dataset/with_neighbor/surrounding/100000_seeds/train/train_data_tilesize_64_64.h5 --val-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/Isotropic_1024/PT_Dataset/with_neighbor/surrounding/100000_seeds/val/val_data_tilesize_64_64.h5 --model-dir /net/scratch/zhuokai/Memory-PIVnet/model/Isotropic_1024/100000_seeds/time_span_5/ --output-dir /net/scratch/zhuokai/Memory-PIVnet/figs/Isotropic_1024/100000_seeds/time_span_5/ --checkpoint-dir /net/scratch/zhuokai/Memory-PIVnet/model/Isotropic_1024/100000_seeds/time_span_5/memory-piv-net_multi-frame_5_batch4_epoch5.pt -t 5 -l RMSE --data-type multi-frame --batch-size 4 --num-epoch 2 -v
