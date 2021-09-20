# The script verifies if the ground truth saved during inference is the same as original
import os
from tqdm import tqdm
import numpy as np

ground_truth_dir = '/home/zhuokai/Desktop/nvme1n1p1/Data/LMSI/Zhao_JHTDB/Isotropic_1024/Figs/test/z_662_762/50000/'
recovered_ground_truth_dir = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/output/Isotropic_1024/velocity/amnesia_memory/50000_seeds/no_pe/time_span_5/true_vel_field/'

# load all the files
for i in tqdm(range(252)):
    # load ground truth
    ground_truth_path = os.path.join(ground_truth_dir, f'4D_piv_vel_{str(i).zfill(4)}.npy')
    ground_truth = np.load(ground_truth_path)[:, :, :2]

    # load recovered ground truth
    recovered_ground_truth_path = os.path.join(recovered_ground_truth_dir, f'true_velocity_{i}.npz')
    recovered_truth = np.load(recovered_ground_truth_path)['velocity']

    if not np.allclose(ground_truth, recovered_truth):
        raise Exception(f'Results vary at index {i}')

print(f'Recovered field is the same as original field')
