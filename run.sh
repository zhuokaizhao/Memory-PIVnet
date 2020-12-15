#!/bin/bash
# for i in `seq 10 50`;
COUNTER=0
while [  $COUNTER -lt 10 ]; do
	echo "Training epoch" $i
	if [ $i = 1 ]; then
		echo "#!/bin/bash" >> job${i}.slurm
		echo "#SBATCH --job-name=train" >> job${i}.slurm
		echo "#SBATCH --nodes=1" >> job${i}.slurm
		echo "#SBATCH --ntasks=1" >> job${i}.slurm
		echo "#SBATCH --cpus-per-task=1" >> job${i}.slurm
		echo "#SBATCH --mem=8G" >> job${i}.slurm
		echo "#SBATCH --partition general" >> job${i}.slurm
		echo "#SBATCH --gres=gpu:rtx8000:1" >> job${i}.slurm
		echo "#SBATCH --time=04:00:00" >> job${i}.slurm
		echo "#SBATCH --mail-type=begin" >> job${i}.slurm
		echo "#SBATCH --mail-type=end" >> job${i}.slurm
		echo "#SBATCH --mail-user=zhuokai@cs.uchicago.edu" >> job${i}.slurm
		echo "conda activate pt-env" >> job${i}.slurm
		echo "srun python /net/scratch/zhuokai/Memory-PIVnet/main.py --mode train --network-model memory-piv-net --train-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/Isotropic_1024/PT_Dataset/with_neighbor/surrounding/100000_seeds/train/train_data_tilesize_64_64.h5 --val-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/Isotropic_1024/PT_Dataset/with_neighbor/surrounding/100000_seeds/val/val_data_tilesize_64_64.h5 --model-dir /net/scratch/zhuokai/Memory-PIVnet/model/Isotropic_1024/100000_seeds/time_span_5/ --output-dir /net/scratch/zhuokai/Memory-PIVnet/figs/Isotropic_1024/100000_seeds/time_span_5/ -t 5 -l RMSE --data-type multi-frame --batch-size 4 --num-epoch 1 -v" >> job${i}.slurm
	else
		prev=$((i-1))
		echo "#!/bin/bash" >> job${i}.slurm
		echo "#SBATCH --job-name=train" >> job${i}.slurm
		echo "#SBATCH --nodes=1" >> job${i}.slurm
		echo "#SBATCH --ntasks=1" >> job${i}.slurm
		echo "#SBATCH --cpus-per-task=1" >> job${i}.slurm
		echo "#SBATCH --mem=8G" >> job${i}.slurm
		echo "#SBATCH --partition general" >> job${i}.slurm
		echo "#SBATCH --gres=gpu:rtx8000:1" >> job${i}.slurm
		echo "#SBATCH --time=04:00:00" >> job${i}.slurm
		echo "#SBATCH --mail-type=begin" >> job${i}.slurm
		echo "#SBATCH --mail-type=end" >> job${i}.slurm
		echo "#SBATCH --mail-user=zhuokai@cs.uchicago.edu" >> job${i}.slurm
		echo "conda activate pt-env" >> job${i}.slurm
	    echo "srun python /net/scratch/zhuokai/Memory-PIVnet/main.py --mode train --network-model memory-piv-net --train-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/Isotropic_1024/PT_Dataset/with_neighbor/surrounding/100000_seeds/train/train_data_tilesize_64_64.h5 --val-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/Isotropic_1024/PT_Dataset/with_neighbor/surrounding/100000_seeds/val/val_data_tilesize_64_64.h5 --model-dir /net/scratch/zhuokai/Memory-PIVnet/model/Isotropic_1024/100000_seeds/time_span_5/ --output-dir /net/scratch/zhuokai/Memory-PIVnet/figs/Isotropic_1024/100000_seeds/time_span_5/ --checkpoint-dir /net/scratch/zhuokai/Memory-PIVnet/model/Isotropic_1024/100000_seeds/time_span_5/memory-piv-net_multi-frame_5_batch4_epoch${prev}.pt -t 5 -l RMSE --data-type multi-frame --batch-size 4 --num-epoch 1 -v" >> job${i}.slurm
	fi
	let COUNTER=COUNTER+1
	# sbatch job${i}.slurm
	# sleep 2h
done