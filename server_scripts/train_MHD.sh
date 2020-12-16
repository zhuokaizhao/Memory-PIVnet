#!/bin/bash
# ask for starting and ending epoch
echo "What GPU do you want to use (gpu:1 or gpu:rtx8000:1)?"
read GPU
echo "What is the number of seeds (10000, 50000 or 100000)?"
read num_seeds
echo "What is the time span (3, 5, 7 or 9)?"
read time_span
echo "What is the first epoch (inclusive)?"
read start_epoch
echo "What is the last epoch (inclusive)?"
read end_epoch

for i in `seq $start_epoch $end_epoch`; do
    echo "Training epoch" $i
	echo "#!/bin/bash" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --job-name=train_MHD_${num_seeds}_${time_span}_${i}" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --nodes=1" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --ntasks=1" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --cpus-per-task=1" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --mem=15G" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --partition general" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --gres=${GPU}" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --time=02:00:00" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --mail-type=begin" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --mail-type=end" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --mail-user=zhuokai@cs.uchicago.edu" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	# echo "conda activate pt-env" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	if [ $i = 1 ]; then
		echo "srun python /net/scratch/zhuokai/Memory-PIVnet/main.py --mode train --network-model memory-piv-net --train-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/MHD_1024/PT_Dataset/${num_seeds}_seeds/train/train_data_tilesize_64_64.h5 --val-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/MHD_1024/PT_Dataset/${num_seeds}_seeds/val/val_data_tilesize_64_64.h5 --model-dir /net/scratch/zhuokai/Memory-PIVnet/model/MHD_1024/${num_seeds}_seeds/time_span_${time_span}/ --output-dir /net/scratch/zhuokai/Memory-PIVnet/figs/MHD_1024/${num_seeds}_seeds/time_span_${time_span}/ -t ${time_span} -l RMSE --data-type multi-frame --batch-size 4 --num-epoch 1 -v" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	else
		prev=$((i-1))
	    echo "srun python /net/scratch/zhuokai/Memory-PIVnet/main.py --mode train --network-model memory-piv-net --train-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/MHD_1024/PT_Dataset/${num_seeds}_seeds/train/train_data_tilesize_64_64.h5 --val-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/MHD_1024/PT_Dataset/${num_seeds}_seeds/val/val_data_tilesize_64_64.h5 --model-dir /net/scratch/zhuokai/Memory-PIVnet/model/MHD_1024/${num_seeds}_seeds/time_span_${time_span}/ --output-dir /net/scratch/zhuokai/Memory-PIVnet/figs/MHD_1024/${num_seeds}_seeds/time_span_${time_span}/ --checkpoint-dir /net/scratch/zhuokai/Memory-PIVnet/model/MHD_1024/${num_seeds}_seeds/time_span_${time_span}/memory-piv-net_multi-frame_${time_span}_batch4_epoch${prev}.pt -t ${time_span} -l RMSE --data-type multi-frame --batch-size 4 --num-epoch 1 -v" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	fi
	# submit job
    sbatch train_MHD_${num_seeds}_${time_span}_${i}.slurm
	rm train_MHD_${num_seeds}_${time_span}_${i}.slurm
	# wait until the current epoch training finished
	until [ -f /net/scratch/zhuokai/Memory-PIVnet/model/MHD_1024/${num_seeds}_seeds/time_span_${time_span}/memory-piv-net_multi-frame_${time_span}_batch4_epoch${i}.pt ]
	do
		sleep 5
	done
done

echo "Training from epoch $start_epoch to $end_epoch is completed"
exit