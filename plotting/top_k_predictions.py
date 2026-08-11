import logging

from matplotlib import pyplot as plt
from torchvision.transforms import v2

from plotting.simple_plots import stackRGB
from utils.extra import open_file_as_tensor, open_file_as_tensor_norm

def plot_top_k_predictions(df, config, path=None, save=True):
    """Plots top-k predictions from a given prediction dataframe"""
    # Store rgbs
    rgbs = []

    # Get labels for tiles
    pos_mask = df['Match_label'].astype(bool).tolist()[1:]
    labels = (["Orginal PS tile"] + [f"The actual Match.\n Similarity: {df.iloc[0, -2]:.3f}"] + [ f"Top-{i + 1}: {str(mask)} Match.\n Similarity: {df.iloc[i + 1, -2]:.3f}" for i, mask in enumerate(pos_mask)])

    # Get paths to tiles
    PS_path = df.iloc[0, 0]
    S2_paths = list(df.iloc[:, 1])
    k = len(S2_paths) - 1

    # Define normalisation regime
    norm_mode = 'PS' if config.normalise else 'stack'

    # If none, normalise based on PS and S2 min max values within k tiles
    if norm_mode == 'stack':
        ps_tile = open_file_as_tensor(PS_path)[:3]

        min_vals = [ps_tile.min()]
        max_vals = [ps_tile.max()]

        for s2_path in S2_paths:
            s2_tile = open_file_as_tensor(s2_path)[1:4]
            min_vals.append(s2_tile.min())
            max_vals.append(s2_tile.max())

        norm_mode = (sum(min_vals)/len(min_vals), sum(max_vals)/len(max_vals))

    # Open PS tile as RGB
    PS_tensor = open_file_as_tensor_norm(PS_path, norm_mode=norm_mode)
    PS_tensor = v2.CenterCrop(size=(config.tile_size, config.tile_size))(PS_tensor)
    rgbs.append(stackRGB(0, 1, 2, PS_tensor))

    # Change norm mode if not stack
    norm_mode = 'S2' if norm_mode == 'PS' else norm_mode

    # Open all S2 tiles as RGBs
    for S2_path in S2_paths:
        S2_tensor = open_file_as_tensor_norm(S2_path, norm_mode=norm_mode)
        S2_tensor = v2.CenterCrop(size=(config.right_tile_size, config.right_tile_size))(S2_tensor)
        rgbs.append(stackRGB(1, 2, 3, S2_tensor))

    # Plot the grid
    cols = min(k + 2, 5)
    rows = (k + 2) // cols + ((k + 2) % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < k + 2:
            img = rgbs[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(labels[i])

            # Set border color for 2-k grids based on labels
            if i < 2:
                color = None
            elif pos_mask[i - 2]:
                color = 'green'
            else:
                color = 'red'
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(3)
                ax.spines[axis].set_color(color)

            # Remove ticks
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    fig.suptitle(f"Top {k} predictions", fontsize=16)
    plt.tight_layout()

    if save:
        plt.savefig(path)
        logging.info(f'Top-{k} prediction figure saved to {path}.')
        plt.close()
    else:
        plt.show()
