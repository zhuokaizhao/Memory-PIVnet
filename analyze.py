# The script analyzes all output .npy files and generate one figure
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# fig_prefix = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/figs/Isotropic_1024/10000_seeds'
# fig_prefix = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/figs/MHD_1024/velocity/amnesia_memory/10000_seeds'
fig_prefix = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/figs/Isotropic_1024/velocity/'
# time span 0 indicates the piv-lfn-en
# time_spans = ['0', '2', '3', '5', '7', '9']
# mode = ['no_pe', 'pe']
# time_spans = ['5']
mode = ['amnesia_memory', 'non_amnesia_memory']
time_spans = ['0', '3', '5']
# inclusive start and end time stamp
start_t = 0
end_t = 251
loss = 'RMSE'

# create full paths
fig_paths = []
for i in range(len(mode)):
    for j in range(len(time_spans)):
        if time_spans[j] == '0':
            if mode[i] == 'amnesia_memory':
                cur_path = os.path.join('/home/zhuokai/Desktop/UChicago/Research/PIV-LiteFlowNet-en-Pytorch/figs/Isotropic_1024/10000_seeds/21',
                                            f'piv-lfn-en_{start_t}_{end_t-1}_all_losses.npy')
            else:
                continue
        elif time_spans[j] == '2':
            cur_path = os.path.join(fig_prefix, 'image_pair', 'tiled',
                                        f'memory-piv-net-ip-tiled_{time_spans[j]}_{start_t}_{end_t}_all_losses_blend.npy')
        else:
            if mode[i] == 'amnesia_memory':
                cur_path = os.path.join(fig_prefix, f'{mode[i]}', '10000_seeds', 'time_span_'+time_spans[j], 'multiframe_with_neighbor', 'surrounding',
                                            f'memory-piv-net_{time_spans[j]}_{start_t}_{end_t}_all_losses_blend.npy')
            else:
                cur_path = os.path.join(fig_prefix, f'{mode[i]}', '10000_seeds', 'time_span_'+time_spans[j],
                                            f'memory-piv-net_{time_spans[j]}_{start_t}_{end_t}_all_losses_blend.npy')
        print(cur_path)
        fig_paths.append(cur_path)

# fig_paths = []
# for i in range(len(mode)):
#     for j in range(len(time_spans)):
#         cur_path = os.path.join(fig_prefix, f'{mode[i]}', 'time_span_' + time_spans[j],
#                                 f'memory-piv-net_{time_spans[j]}_{start_t}_{end_t}_all_losses_blend.npy')

#         fig_paths.append(cur_path)

# fig_paths.append('/home/zhuokai/Desktop/UChicago/Research/PIV-LiteFlowNet-en-Pytorch/figs/MHD_1024/10000_seeds/piv-lfn-en_0_50_all_losses.npy')
print(len(fig_paths))
print(fig_paths)

# load the numpy arrays
all_losses = []
for i in range(len(fig_paths)):
    print(f'Loading losses from {fig_paths[i]}')
    cur_losses = np.load(fig_paths[i])
    # print(cur_losses)
    # for j in range(len(cur_losses)):
    #     if cur_losses[j] < 0.021:
    #         cur_losses[j] = np.mean([cur_losses[j-1], cur_losses[j+1]])
    print(f'Mean of cur losses is {np.mean(cur_losses)}')
    all_losses.append(cur_losses)

# plot all the loss curves in one plot
t = np.arange(start_t, end_t+1)
fig, ax = plt.subplots()
for i in range(len(all_losses)):
    ax.plot(t[:len(all_losses[i])], all_losses[i])

ax.set(xlabel='timestamp', ylabel=f'{loss}')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.legend(['PLFN-en''MPN-2', 'MPN-3', 'MPN-5', 'MPN-7', 'MPN-9'])
# ax.legend(['MemoryPIVnet', 'MemoryPIVnet+PE', 'PIV-LiteFlowNet-en'])
ax.legend(['PIV-LiteFlowNet-en', 'Amnesia_3', 'Amnesia_5', 'Non-Amnesia_3', 'Non-Amnesia_5'])
loss_curve_path = os.path.join(fig_prefix, 'all_losses_blend.svg')
fig.savefig(loss_curve_path, bbox_inches='tight')

print(f'\nResulting curve plot has been saved to {loss_curve_path}')
