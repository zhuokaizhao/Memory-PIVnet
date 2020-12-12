# The script analyzes all output .npy files and generate one figure
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

fig_prefix = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/figs/Isotropic_1024/10000_seeds'
# time span 0 indicates the piv-lfn-en
time_spans = ['0', '2', '3', '5', '7', '9']
# inclusive start and end time stamp
start_t = 0
end_t = 251
loss = 'RMSE'

# create full paths
fig_paths = []
for i in range(len(time_spans)):
    if time_spans[i] == '0':
        cur_path = os.path.join('/home/zhuokai/Desktop/UChicago/Research/PIV-LiteFlowNet-en-Pytorch/figs/10000_seeds/21',
                                    f'piv-lfn-en_{start_t}_{end_t-1}_all_losses.npy')
    elif time_spans[i] == '2':
        cur_path = os.path.join(fig_prefix, 'image_pair', 'tiled',
                                    f'memory-piv-net-ip-tiled_{time_spans[i]}_{start_t}_{end_t}_all_losses_blend.npy')
    else:
        cur_path = os.path.join(fig_prefix, 'time_span_' + time_spans[i], 'multiframe_with_neighbor', 'surrounding',
                                    f'memory-piv-net_{time_spans[i]}_{start_t}_{end_t}_all_losses_blend.npy')

    fig_paths.append(cur_path)

# load the numpy arrays
all_losses = []
for i in range(len(fig_paths)):
    print(f'Loading losses from {fig_paths[i]}')
    cur_losses = np.load(fig_paths[i])
    print(f'Mean of cur losses is {np.mean(cur_losses)}')
    all_losses.append(cur_losses)

# plot all the loss curves in one plot
t = np.arange(start_t, end_t+1)
fig, ax = plt.subplots()
for i in range(len(all_losses)):
    ax.plot(t[:len(all_losses[i])], all_losses[i])

ax.set(xlabel='timestamp', ylabel=f'{loss}')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(['PLFN-en', 'MPN-2', 'MPN-3', 'MPN-5', 'MPN-7', 'MPN-9'])
loss_curve_path = os.path.join(fig_prefix, 'all_losses_blend.svg')
fig.savefig(loss_curve_path, bbox_inches='tight')

print(f'\nResulting curve plot has been saved to {loss_curve_path}')
