import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsort_keygen
import numpy as np
import logging

plt.style.use('seaborn-v0_8')

"""Creates bar plots for the report"""

def metric_plot(df_sub, col_name, title, save_name=None, y_lim=None, scale=True):
    n = len(df_sub)
    sub = df_sub.loc[:, [col_name, 'name']]

    plt.figure(figsize=(15, 6))

    for i in range(n):
        plt.bar(i, sub[col_name].iloc[i], color=colors[1])
        if scale:
            plt.text(i, sub[col_name].iloc[i] - 0.02, f'{100*sub[col_name].iloc[i]:.1f}', ha='center', fontsize=16, color='black')
        else:
            plt.text(i, sub[col_name].iloc[i] - 0.02, f'{sub[col_name].iloc[i]:.3f}', ha='center', fontsize=16, color='black')

    plt.title(title, fontsize=20)

    plt.xticks(np.arange(n), labels=sub.name, rotation=0, ha='center')

    if y_lim:
        plt.ylim(y_lim)
    plt.tight_layout()

    if save_name:
        plt.savefig(f'output/{save_name}.png')
    else:
        plt.show()
        logging.info(f'Saved {save_name}.png')

def similarity_plot(df_sub, save_name=None):
    mixed_sim = df_sub.test_similarity
    neg_sim = df_sub.test_neg_similarity
    pos_sim = df_sub.test_pos_similarity

    n = len(mixed_sim)

    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=(16, 8))

    for i in range(n):
        # Box plot
        plt.bar(i, pos_sim[i] - neg_sim[i], bottom=neg_sim[i], color=colors[1])

        # Mean line
        plt.plot([i - 0.25, i + 0.25], [mixed_sim[i], mixed_sim[i]], color=colors[0], linewidth=2, label='Mean' if i == 0 else "")

        # Labels
        if abs(neg_sim[i] - pos_sim[i]) < 0.05:
            # plt.text(i, pos_sim[i] + 0.015, f'Neg: {neg_sim[i]:.2f}\nMean: {mixed_sim[i]:.2f}\nPos: {pos_sim[i]:.2f}', ha='center', fontsize=18,color='black')
            plt.text(i, pos_sim[i] + 0.015, f'Neg: {neg_sim[i]:.2f}\nPos: {pos_sim[i]:.2f}', ha='center', fontsize=18,color='black')
        else:
            plt.text(i, neg_sim[i] - 0.02, f'Neg: {neg_sim[i]:.2f}', ha='center', va='center', fontsize=18, color='black')
            plt.text(i, pos_sim[i] + 0.01, f'Pos: {pos_sim[i]:.2f}', ha='center', va='center', fontsize=18, color='black')
            plt.text(i, mixed_sim[i] + 0.015, f'{mixed_sim[i]:.2f}', ha='center', va='center', fontsize=18, color='black')

    # Axes labels and title
    # plt.title('Average Positive, Negative and Mixed Similarities', fontsize=24, weight='bold')

    # Plot the threshold
    threshold_line = plt.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='Classification\nthreshold', zorder=4)
    # plt.xlabel('Batch Size', fontsize=24, fontweight='bold')
    plt.xlabel('Framework Variants', fontsize=24, fontweight='bold')
    plt.ylabel('Average Similarity', fontsize=24, fontweight='bold')
    # plt.ylim((0, 0.8))
    plt.ylim((-0.07, 0.8))

    plt.xticks(np.arange(n), labels=df_sub.name, ha='center', fontsize=20)
    plt.yticks(fontsize=18)

    # Legend mixed sim
    plt.legend(loc='upper right', fontsize=20)
    plt.tight_layout()

    if save_name:
        plt.savefig(f'output/visualise_similarity_range_{save_name}.png')
    else:
        plt.show()

# -------------------------------------------------------------------
colors = ['#f6992a', '#6aade4']
df = pd.read_csv('output/validate_top_5_acc_model_performance.csv')

df = df.sort_values(
    by="run_name",
    key=natsort_keygen()
)

dic_names = {'proposed_200ep': "Double\nAcquisition \nNN (200ep)",
            'proposed': "Double \nAcquisition \nNN (800ep\nthe best)",
            'drsl': "DRSL",
            'siamnn_simple': "SiamNN-\nSimple",
            'siamnn_3_layer': "SiamNN-\n3-layer",
            'siamnn_individual': "SiamNN-\nIndividual",
            'simclr_3_layer' : "SimCLR-\n3-layer \n(Nt-Xent-\nOut)",
            'simclr_individual_nt_xent_out': "SimCLR-\nIndividual \n(Nt-Xent-\nOut)",
            'simclr_individual_nt_xent_in': "SimCLR-\nIndividual \n(Nt-Xent-\nIn)",
            'simclr_individual_nt_xent_full': "SimCLR-\nIndividual \n(Nt-Xent-\nFull)",
            'aug_mild': "Mild",
            'aug_medium': "Medium",
            'aug_harsh': "Harsh",
            'aug_shuffle': "Shuffled",
            'resnet50_moco': "Resnet50",
             'resnet50_dino': "Resnet50 (DINo)",
             'moco_s2_only': 'MoCo for S2 only'
             }

# Similiraty ranges for configurations
experiments = ['proposed_200ep', 'drsl', 'siamnn_simple', 'siamnn_3_layer', 'siamnn_individual', 'simclr_3_layer', 'simclr_individual_nt_xent_out', 'simclr_individual_nt_xent_in', 'simclr_individual_nt_xent_full']

df_sub = df[df.run_name.isin(experiments)]
df_sub['name'] = df_sub['run_name'].apply(lambda x: dic_names.get(x, x))
df_sub['run_name'] = pd.Categorical(df_sub['run_name'], categories=experiments, ordered=True)
df_sub = df_sub.sort_values('run_name', ignore_index=True)


similarity_plot(df_sub)
similarity_plot(df_sub, 'similarity_plot')
metric_plot(df_sub, 'test_acc', "Accuracy", y_lim=(0.45, 0.91))


df_sub_k = pd.read_csv('output/predictions/top_k_means.csv')
df_sub_k = df_sub_k[df_sub_k.run_name.isin(experiments)]
df_sub_k['name'] = df_sub_k['run_name'].apply(lambda x: dic_names.get(x, x))
df_sub_k['run_name'] = pd.Categorical(df_sub_k['run_name'], categories=experiments, ordered=True)
df_sub_k = df_sub_k.sort_values('run_name', ignore_index=True)

for k in [1, 3, 5, 10, 50]:
    metric_plot(df_sub_k, f'{k}', f"Top-{k} Accuracy")

metric_plot(df_sub_k, 'mean_position', "Mean Position", scale=False)
# -------------------------------------------------------------------
# Similiraty ranges for batch sizes
dic_names = {'proposed_200ep':  "128 (200ep)",
             'batch16_200ep': "16 (200ep)",
             'batch32_200ep': "32 (200ep)",
             'batch64_200ep': "64 (200ep)",
             'batch64_400ep': "64 (400ep)",
             'batch_256_200ep': "256 (200ep)",
             'batch256_1600ep': "256 (1600ep)",
             }

experiments = ['batch16_200ep', 'batch32_200ep', 'batch64_200ep', 'proposed_200ep',   'batch_256_200ep']

df_sub = df[df.run_name.isin(experiments)]
df_sub['name'] = df_sub['run_name'].apply(lambda x: dic_names.get(x, x))
df_sub['run_name'] = pd.Categorical(df_sub['run_name'], categories=experiments, ordered=True)
df_sub = df_sub.sort_values('run_name', ignore_index=True)

similarity_plot(df_sub)
similarity_plot(df_sub, save_name="batch")
metric_plot(df_sub, 'test_acc', "Accuracy")

df_sub_k = pd.read_csv('output/predictions/top_k_means.csv')
df_sub_k = df_sub_k[df_sub_k.run_name.isin(experiments)]
df_sub_k['name'] = df_sub_k['run_name'].apply(lambda x: dic_names.get(x, x))
df_sub_k['run_name'] = pd.Categorical(df_sub_k['run_name'], categories=experiments, ordered=True)
df_sub_k = df_sub_k.sort_values('run_name', ignore_index=True)

for k in [1, 3, 5, 10, 50]:
    metric_plot(df_sub_k, f'{k}', f"Top-{k} Accuracy")

metric_plot(df_sub_k, 'mean_position', "Mean Position", scale=False)