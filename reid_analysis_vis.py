import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def several_in_one_figure(val, name: list=['number of samples', 'mAP', 'rank-1'],
        x_ticks: list = [0, 1, 2, 3], fig_name: str = 'sample.png', x_name: str='vis'):
    fig, ax1 = plt.subplots()
    plt.xticks(rotation=90)
    plt.xlabel(x_name)
    ax2 = ax1.twinx()

    for v, n in zip(val, name):
        x = np.arange(len(v))
        if n == 'number of samples':
            ax1.bar(x, v, color='red', alpha=0.1)
            ax1.tick_params(axis='y')
            ax1.set_ylabel('Number of samples')
        else:
            ax2.plot(x, v)
            ax2.tick_params(axis='y')
    plt.xticks(x-0.5, x_ticks)
    plt.legend(name)
    plt.savefig(fig_name)

if __name__ == "__main__":
    # vis
    vis = True
    if vis:
        data = pd.read_csv('../AllReID_analysis_vis.csv', sep=';')
        data = data.drop([0, 11])
        print(data.columns)
        num_samps = data['number of samples']
        mAP = data['mAP']
        rank_1 = data['rank-1']

        val = [num_samps, mAP, rank_1]
        x_ticks = data['visibility threshold']
        
        several_in_one_figure(val, x_ticks=x_ticks, fig_name='visibility.png', x_name='visibility')

    size = True
    if size:
        data = pd.read_csv('../AllReID_analysis_size.csv', sep=';')
        data = data.drop([0])
        print(data.columns)
        num_samps = data['number of samples']
        mAP = data['mAP']
        rank_1 = data['rank-1']

        val = [num_samps, mAP, rank_1]
        x_ticks = data['size threshold']
        
        several_in_one_figure(val, x_ticks=x_ticks, fig_name='size.png', x_name='bb height')

