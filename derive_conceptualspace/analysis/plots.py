from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

def scatter_2d(df, category, catnames):
    fig, ax = plt.subplots(figsize=(16, 10))
    sp = sns.scatterplot(
        x="x", y="y",
        hue=category,
        palette=sns.color_palette("bright", len(df[category].unique())),
        data=df,
        legend="full",
        alpha=0.7,
        ax=ax
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, catnames.values(), loc=0, title=category.capitalize())
    plt.show()


def scatter_3d(df, category, catnames):
    # fig = px.scatter_3d(df, x='tsne_1', y='tsne_2', z='tsne_3', color='FB_long', opacity=0.7)#, size=[2]*len(df))
    fig = go.Figure(layout=go.Layout(
            scene=dict(camera=dict(eye=dict(x=1, y=1, z=1)), aspectmode="data"),
            autosize=True,
            width=1120,
            height=800,
            margin=dict(l=10, r=10, b=10, t=40, pad=4),
            paper_bgcolor="White",
            title=f"3D-Embedding, colored by {category.capitalize()}"))
    for ncol, part_df in enumerate(set(df[category])):
        fig.add_trace(
            go.Scatter3d(
                name=catnames[part_df],
                mode='markers',
                x=df[df[category] == part_df]["x"],
                y=df[df[category] == part_df]["y"],
                z=df[df[category] == part_df]["z"],
                marker=dict(
                    color=ncol,
                    size=1.5,
                    line=dict(
                        width=0
                    )
                ),
            )
        )
    # fig.update_layout(showlegend=False)
    fig.update_layout(legend={'itemsizing': 'constant'})
    fig.update_layout(legend_font_size=16, title_font_size=20)
    fig.show()