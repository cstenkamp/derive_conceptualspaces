import warnings
from datetime import datetime
import os
import threading
import sys

import numpy as np
import matplotlib.pyplot as plt

from misc_util.pretty_print import isnotebook

def show_fig(fig, title):
    #TODO maybe be able to overwrite this with an env-var
    is_pycharm = "PYCHARM_HOSTED" in os.environ
    if is_pycharm and not isnotebook():
        fig.show()
    if not isnotebook():
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
    max_val = x.max() if not cutoff_percentile else round(np.percentile(x, cutoff_percentile)) + 1
    old_max = x.max()
    x[x >= max_val] = max_val
    q25, q75 = np.percentile(x, [25, 75])
    if q75 > q25:
        bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
        bins = round((x.max() - x.min()) / bin_width)
        bins = min(bins, (x.max() - x.min()))
        kwargs["bins"] = round(bins)
    elif x.max() - x.min() < 30:
        bins = x.max() - x.min()
        kwargs["bins"] = round(bins)
    full_data = dict(type="hist", x=x, kwargs=kwargs, xlim=(0, max_val), cutoff_percentile=cutoff_percentile, xlabel=xlabel,
                     ylabel=ylabel, title=title, max_val=max_val, old_max=old_max)
    prepare_fig(full_data, title)


def prepare_fig(full_data, title):
    serialize_plot(title, full_data)
    if threading.current_thread() is not threading.main_thread():
        #mpl needs main-thread, snakemake often is not mainthread!
        warnings.warn("Cannot plot the figure as we are not in the main-thread!")
        return
    actually_plot(full_data)


def actually_plot(full_data):
    fig, ax = plt.subplots()
    if full_data["type"] == "hist":
        ax.hist(full_data["x"], **full_data["kwargs"])
        ax.set_xlim(*full_data["xlim"])
        if full_data.get("cutoff_percentile") is not None:
            ax.set_xticks(list(plt.xticks()[0][:-1]) + [full_data["max_val"]])
            ax.set_xticklabels([str(round(i)) for i in plt.xticks()[0]][:-1] + [f"{full_data['max_val']}-{full_data['old_max']}"], ha="right", rotation=45)
        ax.set_ylabel(full_data["ylabel"])
        ax.set_xlabel(full_data["xlabel"])
    else:
        raise NotImplementedError(f"Cannot do plot of type {full_data['type']}!")
    if full_data["title"]: plt.title(full_data["title"])
    plt.tight_layout()
    show_fig(fig, full_data["title"])


def serialize_plot(title, full_data):
    if hasattr(sys.stdout, "ctx"): #TODO getting the json_serializer this way is dirty as fuck!
        sys.stdout.ctx.obj["json_persister"].add_plot(title, full_data)


if __name__ == "__main__":
    show_hist([1, 2, 3, 1, 2, 3, 4, 1, 2], "simple plot")