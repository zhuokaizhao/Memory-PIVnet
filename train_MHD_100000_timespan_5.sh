#!/bin/bash
# ask for starting and ending epoch
echo "What is the first epoch (inclusive)?"
read start_epoch
echo "What is the last epoch (inclusive)?"
read end_epoch

for i in `seq $start_epoch $end_epoch`; do
    echo "Training epoch" $i
	echo "#!/bin/bash" >> train_MHD_100000_5_${i}.slurm
	echo "#SBATCH --job-name=train_MHD_100000_5" >> train_MHD_100000_5_${i}.slurm
	echo "#SBATCH --nodes=1" >> train_MHD_100000_5_${i}.slurm
	echo "#SBATCH --ntasks=1" >> train_MHD_100000_5_${i}.slurm
	echo "#SBATCH --cpus-per-task=1" >> train_MHD_100000_5_${i}.slurm
	echo "#SBATCH --mem=8G" >> train_MHD_100000_5_${i}.slurm
	echo "#SBATCH --partition general" >> train_MHD_100000_5_${i}.slurm
	echo "#SBATCH --gres=gpu:rtx8000:1" >> train_MHD_100000_5_${i}.slurm
	echo "#SBATCH --time=02:00:00" >> train_MHD_100000_5_${i}.slurm
	echo "#SBATCH --mail-type=begin" >> train_MHD_100000_5_${i}.slurm
	echo "#SBATCH --mail-type=end" >> train_MHD_100000_5_${i}.slurm
	echo "#SBATCH --mail-user=zhuokai@cs.uchicago.edu" >> train_MHD_100000_5_${i}.slurm
	echo "conda activate pt-env" >> train_MHD_100000_5_${i}.slurm
	if [ $i = 1 ]; then
		echo "srun python /net/scratch/zhuokai/Memory-PIVnet/main.py --mode train --network-model memory-piv-net --train-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/MHD_1024/PT_Dataset/100000_seeds/train/train_data_tilesize_64_64.h5 --val-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/MHD_1024/100000_seeds/val/val_data_tilesize_64_64.h5 --model-dir /net/scratch/zhuokai/Memory-PIVnet/model/MHD_1024/100000_seeds/time_span_5/ --output-dir /net/scratch/zhuokai/Memory-PIVnet/figs/MHD_1024/100000_seeds/time_span_5/ -t 5 -l RMSE --data-type multi-frame --batch-size 4 --num-epoch 1 -v" >> train_MHD_100000_5_${i}.slurm
	else
		prev=$((i-1))
	    echo "srun python /net/scratch/zhuokai/Memory-PIVnet/main.py --mode train --network-model memory-piv-net --train-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/MHD_1024/PT_Dataset/100000_seeds/train/train_data_tilesize_64_64.h5 --val-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/MHD_1024/100000_seeds/val/val_data_tilesize_64_64.h5 --model-dir /net/scratch/zhuokai/Memory-PIVnet/model/MHD_1024/100000_seeds/time_span_5/ --output-dir /net/scratch/zhuokai/Memory-PIVnet/figs/MHD_1024/100000_seeds/time_span_5/ --checkpoint-dir /net/scratch/zhuokai/Memory-PIVnet/model/MHD_1024/100000_seeds/time_span_5/memory-piv-net_multi-frame_5_batch4_epoch${prev}.pt -t 5 -l RMSE --data-type multi-frame --batch-size 4 --num-epoch 1 -v" >> train_MHD_100000_5_${i}.slurm
	fi
	# submit job
    sbatch train_MHD_100000_5_${i}.slurm
	rm train_MHD_100000_5_${i}.slurm
	# wait until the current epoch training finished
	until [ -f /net/scratch/zhuokai/Memory-PIVnet/model/MHD_1024/100000_seeds/time_span_5/memory-piv-net_multi-frame_5_batch4_epoch${i}.pt ]
	do
		sleep 5
	done
done

echo "Training from epoch $start_epoch to $end_epoch is completed"
exit