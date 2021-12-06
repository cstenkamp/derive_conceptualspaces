from datetime import datetime
import os

import numpy as np
import matplotlib.pyplot as plt

def show_fig(fig, title):
    #TODO maybe be able to overwrite this with an env-var
    is_pycharm = "PYCHARM_HOSTED" in os.environ
    if is_pycharm:
        fig.show()
    title = title.replace(" ", "_")
    title += "_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.getenv("MA_BASE_DIR") or os.environ["HOME"]
    save_path = os.path.join(save_path, "saved_plots", datetime.now().strftime("%Y-%m-%d"), title+".png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving figure `{title}` under `{save_path}`")
    fig.savefig(save_path)

def show_hist(x, title="", xlabel="Data", ylabel="Count", cutoff_percentile=95, **kwargs): # density=False shows counts
    #see https://stackoverflow.com/a/33203848/5122790
    #Freedman–Diaconis number of bins
    x = np.array(x)
    max_val = round(np.percentile(x, cutoff_percentile)) + 1
    old_max = x.max()
    x[x >= max_val] = max_val
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round((x.max() - x.min()) / bin_width)
    bins = min(bins, (x.max() - x.min()))
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins, **kwargs)
    ax.set_xlim(0, max_val)
    ax.set_xticks(list(plt.xticks()[0][:-1]) + [max_val])
    ax.set_xticklabels([str(round(i)) for i in plt.xticks()[0]][:-1] + [f"{max_val}-{old_max}"], ha="right", rotation=45)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title: plt.title(title)
    plt.tight_layout()
    show_fig(fig, title)


if __name__ == "__main__":
    show_hist([1, 2, 3, 1, 2, 3, 4, 1, 2], "simple plot")