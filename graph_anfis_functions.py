from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.pyplot import cm

from models.anfis import ANFIS

def plot_anfis_rules(anfis_model):
    n_values = 1_000
    fig = plt.figure(figsize=(10,10))
    colors = iter(cm.tab20(np.linspace(0, 1, anfis_model.n_rules)))

    if anfis_model.membership_type == "Gaussian":
        centers = anfis_model.centers.clone().detach().cpu().numpy()
        widths = anfis_model.widths.clone().detach().cpu().numpy()

        min_idx = np.unravel_index(np.argmin(centers), centers.shape)
        max_idx = np.unravel_index(np.argmax(centers), centers.shape)
        ex_min = np.exp(-(centers[min_idx] / (2 * widths[min_idx]))**2) - 10
        ex_max = np.exp(-(centers[max_idx] / (2 * widths[max_idx]))**2) + 10

        x = np.linspace(ex_min, ex_max, n_values)
        for idx, (c, w) in enumerate(zip(centers, widths)):
            color = next(colors)
            for idx_minor, (center, width) in enumerate(zip(c, w)):
                y = np.exp(-((x - center)**2 / (2 * width**2)))
                plt.plot(x, y, c=color, label=f"output_rule: {idx}" if idx_minor == 0 else None)


    elif anfis_model.membership_type == "Triangular":
        centers = anfis_model.centers.clone().detach().cpu().numpy()
        left_widths = anfis_model.left_widths.clone().detach().cpu().numpy()
        right_widths = anfis_model.left_widths.clone().detach().cpu().numpy()

        lefts = centers - left_widths
        rights = centers + right_widths

        min_idx = np.unravel_index(np.argmin(lefts), lefts.shape)
        max_idx = np.unravel_index(np.argmax(rights), rights.shape)
        x_min = lefts[min_idx] - 1
        x_max = rights[max_idx] + 1

        x = np.linspace(x_min, x_max, n_values)
        for idx, (c, l, r) in enumerate(zip(centers, left_widths, right_widths)):
            color = next(colors)
            for idx_minor, (center, left, right) in enumerate(zip(c, l, r)):
                y = np.zeros_like(x)
                left = center - left
                right = center + right

                left_mask = (x >= left) & (x < center)
                right_mask = (x >= center) & (x <= right)
                y[left_mask] = (x[left_mask] - left) / (center - left)
                y[right_mask] = (right - x[right_mask]) / (right - center)
                plt.plot(x, y, c=color, label=f"output_rule: {idx}" if idx_minor == 0 else None)
    else:
        raise NotImplementedError("Plotting of other rule bases is not yet supported")
        return -1

    plt.legend()
    plt.show()

    return 0
