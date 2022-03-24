from textwrap import shorten

from matplotlib import pyplot as plt
import seaborn as sns

from derive_conceptualspace.util.threedfigure import ThreeDFigure


def set_seaborn():
    sns.set(font_scale=1.8)
    sns.set_style("whitegrid")

def scatter_2d(df, category, catnames=None, legend_below=True, legend_cols=2, **kwargs):
    palette = lambda num: sns.color_palette([i for i in sns.color_palette("bright") if not i[0]==i[1]==i[2]], num) #no gray
    fig, ax = plt.subplots(figsize=(16, 10))
    sp = sns.scatterplot(
        x="x", y="y",
        hue=category,
        palette=dict(unknown="gray"),
        data=df[df[category] == "unknown"],
        legend="full",
        alpha=0.4,
        ax=ax
    )
    sp.set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None, title=f"t-SNE 2D-Embedding, colored by {category.capitalize()}")
    sp = sns.scatterplot(
        x="x", y="y",
        hue=category,
        palette=palette(len(df[df[category] != "unknown"][category].unique())),
        data=df[df[category] != "unknown"],
        legend="full",
        alpha=0.7,
        ax=ax,
        **kwargs,
    )
    sp.set(xticklabels=[], xlabel=None, yticklabels=[], ylabel=None, title=f"t-SNE 2D-Embedding, colored by {category.capitalize()}")

    handles, labels = ax.get_legend_handles_labels()
    for h in handles:
        h._sizes = [300]
    legendlabels = catnames.values() if catnames else labels
    if legend_below:
        ax.legend(handles, legendlabels, bbox_to_anchor=(0, -0.01), ncol=legend_cols, loc=2, borderaxespad=0., title=category.capitalize())
    else:
        ax.legend(handles, legendlabels, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., title=category.capitalize())
    ax.grid(False)
    plt.show()
    return fig


def scatter_3d(df, category, catnames=None, name=None, descriptions=None):
    # fig = px.scatter_3d(df, x='tsne_1', y='tsne_2', z='tsne_3', color='FB_long', opacity=0.7)#, size=[2]*len(df))
    name = name or f"3D-Embedding, colored by {category.capitalize()}"
    with ThreeDFigure(width=1120, name=name, bigfont=True) as fig:
        for ncol, part_df in enumerate(set(df[category])):
            if descriptions is not None:
                descs = [descriptions._descriptions[i] for i in list(df[df[category] == part_df].index)]
                custom_data = [{"Name": desc.title, "V.Nr.": "|".join(eval(desc._additionals["veranstaltungsnummer"])),
                    "Class": catnames[df[df[category] == part_df].iloc[n][category]] if catnames else df[df[category] == part_df].iloc[n][category],  "extra": {"Description":shorten(desc.text, 200) }} for n, desc in enumerate(descs)]
            fig.add_markers(df[df[category] == part_df][["x", "y", "z"]].values, name=catnames[part_df] if catnames else part_df, color=ncol, size=1.5, custom_data=custom_data)
        fig.show()
        return fig