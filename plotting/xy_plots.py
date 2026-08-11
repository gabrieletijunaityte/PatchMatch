import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

"""Plots for radius searchand catastrophic forgetting plots"""

def radius_vs_top_k_plot(run_name, study_site, show=False, legend=True):
    colors = ['#f6992a', '#6aade4', '#E04B28FF']

    df = pd.read_csv(f'output/predictions/top_k_means_radius_{study_site}.csv')
    df = df[df.run_name == run_name]

    study_site = study_site.split('_')[0]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['radius'], df['5']*100, label='Top-5', marker='o', color=colors[0], linewidth=3)
    plt.plot(df['radius'], df['3']*100, label='Top-3', marker='o', color=colors[1], linewidth=3)
    plt.plot(df['radius'], df['1']*100, label='Top-1', marker='o', color=colors[2], linewidth=3)

    plt.xlabel('Drift speed (m/s)', fontsize=20, weight='bold')
    plt.ylabel('Top-k accuracy (%)', fontsize=20, weight='bold')
    plt.ylim([0,100])
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    if legend:
        plt.legend(fontsize=20)
    plt.grid(True)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(f'output/figures/{run_name}/top_k_means_radius_{study_site}.png')

radius_vs_top_k_plot('proposed', 'Accra_2', True)
radius_vs_top_k_plot('proposed', 'Marmara', True)


radius_vs_top_k_plot('proposed', 'Accra_2')
radius_vs_top_k_plot('proposed', 'Marmara', legend=False)


def catastrophic_forgetting_plot(show=False):
    colors = ['#f6992a', '#6aade4', '#E04B28FF']

    df = pd.read_csv('output/predictions/top_k_means.csv')

    dic_names = {
            'proposed_200ep': 0,
            'refined_1ep': 1,
            'refined_3ep': 3,
            'refined_5ep': 5,
            'refined_10ep': 10,
            'refined_15ep': 15,
            'refined_20ep': 20,
            'refined_25ep': 25,
            'refined_best': 50
    }

    experiments = dic_names.keys()

    df = df[df.run_name.isin(experiments)]
    df['epoch'] = df['run_name'].map(dic_names)
    df.sort_values('epoch', inplace=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['5']*100, label='Top-5', marker='o', color=colors[0], linewidth=3)
    plt.plot(df['epoch'], df['3']*100, label='Top-3', marker='o', color=colors[1], linewidth=3)
    plt.plot(df['epoch'], df['1']*100, label='Top-1', marker='o', color=colors[2], linewidth=3)

    dic_names['setup_x_1'] = 'Highest\nTop-3\naccuracy'
    df['epoch_lab'] = df['run_name'].map(dic_names)
    plt.xticks(df['epoch'], df['epoch_lab'], fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel('N˚ of SimCLR Pre-Training Epoch', fontsize=20, weight='bold')
    plt.ylabel('Global Top-k Accuracy (%)', fontsize=20, weight='bold')

    plt.legend(fontsize=20)
    plt.grid(True)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(f'output/figures/catastrophic_forgetting.png')

catastrophic_forgetting_plot(True)
catastrophic_forgetting_plot()
