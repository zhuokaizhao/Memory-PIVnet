#!/bin/bash
# ask for starting and ending epoch
# echo "What GPU do you want to use (gpu:1 or gpu:rtx2080ti:1 or gpu:rtx8000:1)?"
# read GPU
echo "What is the number of seeds (10000, 50000 or 100000)?"
read num_seeds
echo "What is the time span (3, 5, 7 or 9)?"
read time_span
echo "What is the first epoch (inclusive)?"
read start_epoch
echo "What is the last epoch (inclusive)?"
read end_epoch

if [ $time_span = 3 ]; then
	GPU="gpu:rtx2080ti:1"
else
	GPU="gpu:a40:1"
fi

for i in `seq $start_epoch $end_epoch`; do
    echo "Training epoch" $i
	echo "#!/bin/bash" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --job-name=train_Isotropic_${num_seeds}_${time_span}_${i}" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --nodes=1" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --ntasks=1" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --cpus-per-task=1" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	# echo "#SBATCH --mem=30G" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --partition general" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --gres=${GPU}" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --time=04:00:00" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --mail-type=begin" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --mail-type=end" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	echo "#SBATCH --mail-user=zhuokai@cs.uchicago.edu" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	# echo "conda activate pt-env" >> train_MHD_${num_seeds}_${time_span}_${i}.slurm
	if [ $i = 1 ]; then
		echo "srun python /net/scratch/zhuokai/Memory-PIVnet/main.py --mode train --network-model memory-piv-net --train-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/Isotropic_1024/PT_Dataset/velocity/${num_seeds}_seeds/train/train_velocity_data_tilesize_64_64.h5 --val-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/Isotropic_1024/PT_Dataset/velocity/${num_seeds}_seeds/val/val_data_tilesize_64_64.h5 --model-dir /net/scratch/zhuokai/Memory-PIVnet/model/Isotropic_1024/velocity/amnesia_memory/${num_seeds}_seeds/time_span_${time_span}/ --output-dir /net/scratch/zhuokai/Memory-PIVnet/figs/Isotropic_1024/velocity/amnesia_memory/${num_seeds}_seeds/time_span_${time_span}/ -t ${time_span} -l RMSE --data-type multi-frame --batch-size 8 --num-epoch 1 -v" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	else
		prev=$((i-1))
	    echo "srun python /net/scratch/zhuokai/Memory-PIVnet/main.py --mode train --network-model memory-piv-net --train-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/Isotropic_1024/PT_Dataset/velocity/${num_seeds}_seeds/train/train_velocity_data_tilesize_64_64.h5 --val-dir /net/scratch/zhuokai/Data/LMSI/Zhao_JHTDB/Isotropic_1024/PT_Dataset/velocity/${num_seeds}_seeds/val/val_data_tilesize_64_64.h5 --model-dir /net/scratch/zhuokai/Memory-PIVnet/model/Isotropic_1024/velocity/amnesia_memory/${num_seeds}_seeds/time_span_${time_span}/ --output-dir /net/scratch/zhuokai/Memory-PIVnet/figs/Isotropic_1024/velocity/amnesia_memory/${num_seeds}_seeds/time_span_${time_span}/ --checkpoint-dir /net/scratch/zhuokai/Memory-PIVnet/model/Isotropic_1024/velocity/amnesia_memory/${num_seeds}_seeds/time_span_${time_span}/memory-piv-net_multi-frame_${time_span}_batch8_epoch${prev}.pt -t ${time_span} -l RMSE --data-type multi-frame --batch-size 8 --num-epoch 1 -v" >> train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	fi
	# submit job
    sbatch train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	rm train_Isotropic_${num_seeds}_${time_span}_${i}.slurm
	# wait until the current epoch training finished
	until [ -f /net/scratch/zhuokai/Memory-PIVnet/model/Isotropic_1024/velocity/amnesia_memory/pe/${num_seeds}_seeds/time_span_${time_span}/memory-piv-net_multi-frame_${time_span}_batch4_epoch${i}.pt ]
	do
		sleep 5
	done
done

echo "Training from epoch $start_epoch to $end_epoch is completed"
exit