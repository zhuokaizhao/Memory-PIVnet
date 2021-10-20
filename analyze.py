# The script analyzes all output .npy files and generate one figure (For Master Paper)
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# different use cases
dataset = ['Isotropic_1024', 'MHD_1024', 'Mixing'][0]
seed_density = ['10000_seeds', '50000_seeds', '100000_seeds'][0]
usage = ['all_methods', 'time_windows', 'amnesia_vs_non-amnesia', 'pe_vs_no-pe', 'HS', 'piv-lfn-en', 'un-lfn', 'memory-piv-net'][2]
output_path = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/analyze_figs/'

if usage == 'all_methods':
    path_prefix = '/home/zhuokai/Desktop/UChicago/Research/'
    methods = ['HS', 'piv-lfn-en', 'un-lfn', 'memory-piv-net']

elif usage == 'time_windows':
    path_prefix = '/home/zhuokai/Desktop/UChicago/Research/'
    # time span 0 indicates the piv-lfn-en
    time_spans = ['0', '3', '5', '7', '9']
    methods = ['piv-lfn-en', 'memory-piv-net', 'memory-piv-net', 'memory-piv-net', 'memory-piv-net']

elif usage == 'amnesia_vs_non-amnesia':
    path_prefix = f'/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/figs/{dataset}/velocity/'
    time_spans = ['3', '5', '7']
    mode = ['amnesia_memory', 'non_amnesia_memory']
    methods = ['memory-piv-net', 'memory-piv-net', 'memory-piv-net', 'memory-piv-net', 'memory-piv-net', 'memory-piv-net']

elif usage == 'pe_vs_no-pe':
    path_prefix = f'/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/figs/{dataset}/velocity/amnesia_memory/{seed_density}/'
    mode = ['no_pe', 'pe']
    # time span 0 indicates the piv-lfn-en
    time_spans = ['3', '5', '7']

elif usage == 'HS':
    path_prefix = f'/home/zhuokai/Desktop/UChicago/Research/HornSchunck-Python/figs/{dataset}/'
    datasets = ['10000_seeds', '50000_seeds', '100000_seeds']

elif usage == 'un-lfn':
    path_prefix = f'/home/zhuokai/Desktop/UChicago/Research/UnLiteFlowNet-PIV/figs/Zhao_JHTDB/{dataset}/'
    datasets = ['10000_seeds', '50000_seeds', '100000_seeds']

elif usage == 'memory-piv-net':
    path_prefix = '/home/zhuokai/Desktop/UChicago/Research/Memory-PIVnet/figs/Isotropic_1024/velocity/amnesia_memory/'
    datasets = ['10000_seeds', '50000_seeds', '100000_seeds']


# inclusive start and end time stamp
if dataset == 'Isotropic_1024':
    start_t = 0
    end_t = 251
elif dataset == 'MHD_1024':
    start_t = 0
    end_t = 51
loss = 'RMSE'

# create full paths
result_paths = []
if usage == 'all_methods':
    for i in range(len(methods)):
        if methods[i] == 'HS':
            cur_path = os.path.join(path_prefix, f'HornSchunck-Python/figs/{dataset}/{seed_density}', f'{methods[i]}_{start_t}_{end_t-1}_all_losses.npy')
        elif methods[i] == 'piv-lfn-en':
            cur_path = os.path.join(path_prefix, f'PIV-LiteFlowNet-en-Pytorch/figs/{dataset}/{seed_density}', f'{methods[i]}_{start_t}_{end_t-1}_all_losses.npy')
        elif methods[i] == 'un-lfn':
            cur_path = os.path.join(path_prefix, f'UnLiteFlowNet-PIV/figs/Zhao_JHTDB/{dataset}/{seed_density}', f'{methods[i]}_{start_t}_{end_t-1}_all_losses.npy')
        elif methods[i] == 'memory-piv-net':
            if dataset == 'Isotropic_1024':
                cur_path = os.path.join(path_prefix, f'Memory-PIVnet/figs/{dataset}/velocity/amnesia_memory/{seed_density}/no_pe/time_span_5/', f'{methods[i]}_5_{start_t}_{end_t}_all_losses_blend.npy')
            elif dataset == 'MHD_1024':
                cur_path = os.path.join(path_prefix, f'Memory-PIVnet/figs/{dataset}/velocity/amnesia_memory/{seed_density}/pe/time_span_5/', f'{methods[i]}_5_{start_t}_{end_t}_all_losses_blend.npy')

        result_paths.append(cur_path)

elif usage == 'time_windows':
    for i in range(len(time_spans)):
        if time_spans[i] == '0':
            cur_path = os.path.join(path_prefix, f'PIV-LiteFlowNet-en-Pytorch/figs/{dataset}/{seed_density}', f'{methods[i]}_{start_t}_{end_t-1}_all_losses.npy')
            result_paths.append(cur_path)
        else:
            cur_path = os.path.join(path_prefix, f'Memory-PIVnet/figs/{dataset}/velocity/amnesia_memory/{seed_density}/no_pe', f'time_span_{time_spans[i]}', f'{methods[i]}_{time_spans[i]}_{start_t}_{end_t}_all_losses_blend.npy')
            result_paths.append(cur_path)

elif usage == 'amnesia_vs_non-amnesia':
    for i in range(len(time_spans)):
        for j in range(len(mode)):
            if mode[j] == 'amnesia_memory':
                cur_path = os.path.join(path_prefix, f'{mode[j]}', f'{seed_density}', 'no_pe', f'time_span_{time_spans[i]}',
                                            f'{methods[i]}_{time_spans[i]}_{start_t}_{end_t}_all_losses_blend.npy')
            else:
                cur_path = os.path.join(path_prefix, f'{mode[j]}', f'{seed_density}', f'time_span_{time_spans[i]}',
                                            f'{methods[i]}_{time_spans[i]}_{start_t}_{end_t}_all_losses_blend.npy')
            result_paths.append(cur_path)

elif usage == 'one-sided':
    if time_spans[i] == '2':
        cur_path = os.path.join(path_prefix, f'Memory-PIVnet/figs/{dataset}/velocity/amnesia_memory/{seed_density}/no_pe/image_pair/tiled', f'{methods[i]}_{time_spans[i]}_{start_t}_{end_t}_all_losses.npy')
        result_paths.append(cur_path)

elif usage == 'pe_vs_no-pe':
    for i in range(len(time_spans)):
        if time_spans[i] == '0':
            cur_path = os.path.join(f'/home/zhuokai/Desktop/UChicago/Research/PIV-LiteFlowNet-en-Pytorch/figs/Isotropic_1024/{seed_density}/21',
                                    f'piv-lfn-en_{start_t}_{end_t-1}_all_losses.npy')
            result_paths.append(cur_path)
        else:
            for j in range(len(mode)):
                if mode[j] == 'no_pe':
                    cur_path = os.path.join(path_prefix, f'{mode[j]}', 'time_span_' + time_spans[i], 'multiframe_with_neighbor', 'surrounding',
                                            f'memory-piv-net_{time_spans[i]}_{start_t}_{end_t}_all_losses_blend.npy')
                elif mode[j] == 'pe':
                    cur_path = os.path.join(path_prefix, f'{mode[j]}', 'time_span_' + time_spans[i],
                                            f'memory-piv-net_{time_spans[i]}_{start_t}_{end_t}_all_losses_blend.npy')
                result_paths.append(cur_path)

elif usage == 'HS':
    for i in range(len(datasets)):
        cur_path = os.path.join(path_prefix, datasets[i], f'{usage}_{start_t}_{end_t-1}_all_losses.npy')
        result_paths.append(cur_path)

elif usage == 'memory-piv-net':
    for i in range(len(datasets)):
        cur_path = os.path.join(path_prefix, datasets[i], 'no_pe', 'time_span_5', f'{usage}_5_{start_t}_{end_t}_all_losses.npy')
        result_paths.append(cur_path)

# load the numpy arrays
all_losses = []
for i in range(len(result_paths)):
    print(f'Loading losses from {result_paths[i]}')
    cur_losses = np.load(result_paths[i])
    if dataset == 'Isotropic_1024':
        cur_losses[81] = (cur_losses[80]+cur_losses[82])/2
        cur_losses[154] = (cur_losses[153]+cur_losses[155])/2
    if dataset == 'MHD_1024':
        cur_losses[7] = (cur_losses[6]+cur_losses[8])/2
        cur_losses[21] = (cur_losses[20]+cur_losses[22])/2
        cur_losses[26] = (cur_losses[25]+cur_losses[27])/2
        cur_losses[29] = (cur_losses[28]+cur_losses[30])/2

    print(f'{methods[i]} mean loss is {np.mean(cur_losses)}')

    all_losses.append(cur_losses)

# plot all the loss curves in one plot
t = np.arange(start_t, end_t+1)
fig, ax = plt.subplots()
linestyles = ['-', '--', ':', '-.', '.', '1'][0:len(all_losses)]
for i in range(len(all_losses)):
    ax.plot(t[:len(all_losses[i])], all_losses[i], linestyles[i])

ax.set(xlabel='timestamp', ylabel=f'{loss}')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

if usage == 'all_methods':
    ax.legend(['Horn Schunck', 'PIV-LFN-en', 'Un-LFN', 'Memory-PIVnet'])
    plt.xlabel('time')
    plt.ylabel('RMSE')
elif usage == 'time_windows':
    ax.legend(['PIV-LFN-en', 'MPN-3', 'MPN-5', 'MPN-7', 'MPN-9'])
    plt.xlabel('time')
    plt.ylabel('RMSE')
elif usage == 'amnesia_vs_non-amnesia':
    ax.legend(['Amnesia-3', 'Non-Amnesia-3', 'Amnesia-5', 'Non-Amnesia-5', 'Amnesia-7', 'Non-Amnesia-7'])
    plt.xlabel('time')
    plt.ylabel('RMSE')
elif usage == 'pe_vs_no-pe':
    ax.legend(['Amnesia-3', 'Amnesia+PE-3', 'Amnesia-5', 'Amnesia+PE-5', 'Amnesia-7', 'Amnesia+PE-7'])
    plt.xlabel('time')
    plt.ylabel('RMSE')
elif usage == 'memory-piv-net':
    ax.legend(['Rho=39.68', 'Rho=198.41', 'Rho=396.83'])
    plt.xlabel('time')
    plt.ylabel('RMSE')

loss_curve_path = os.path.join(output_path, f'{usage}_{dataset}_{seed_density}.svg')
fig.savefig(loss_curve_path, bbox_inches='tight')

print(f'\nResulting curve plot has been saved to {loss_curve_path}')
