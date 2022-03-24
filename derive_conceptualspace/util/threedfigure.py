import math
import textwrap

import numpy as np
import plotly.graph_objects as go

from .base_changer import make_base_changer

def make_meshgrid(X=None, minx=None, miny=None, maxx=None, maxy=None, size=None, amount=30, margin=0):
    assert X is not None or (minx is not None and miny is not None and maxx is not None and maxy is not None) or size is not None
    if X is not None:
        minx = min(X[:,0]) if minx is None or min(X[:,0])<minx else minx
        miny = min(X[:,1]) if miny is None or min(X[:,1])<miny else miny
        maxx = max(X[:,0]) if maxx is None or max(X[:,0])>maxx else maxx
        maxy = max(X[:,1]) if maxy is None or max(X[:,1])>maxy else maxy
    elif size is not None and (minx is None and miny is None and maxx is None and maxy is None):
        minx = miny = -size
        maxx = maxy = size
    lsx = np.linspace(minx-margin, maxx+margin, amount)
    lsy = np.linspace(miny-margin, maxy+margin, amount)
    xx, yy = np.meshgrid(lsx,lsy)
    return xx, yy


def ortho_projection_affine(a, b):
    """https://en.wikipedia.org/wiki/Vector_projection"""
    return np.dot(np.dot(a, b) / np.dot(b, b), b)


#TODO: When I de-select a portion of the markers the xlims and ylims shouldn't change (and the viewpoint not update)

class ThreeDFigure():
    hovertemplate = "<br>".join(["X: %{x}", "Y: %{y}", "Z: %{z}"])

    def __init__(self, trafo_fn=None, back_trafo_fn=None, swap_axes=None, name=None, width=1000, height=800, bigfont=False):
        self.trafo_fn = trafo_fn if trafo_fn is not None else lambda x: x
        self.back_trafo_fn = back_trafo_fn if back_trafo_fn is not None else lambda x: x
        self.swap_axes = swap_axes
        # https://community.plotly.com/t/creating-a-3d-scatterplot-with-equal-scale-along-all-axes/15108/7
        self.fig = go.Figure(layout=go.Layout(
            scene=dict(camera=dict(eye=dict(x=1, y=1, z=1)), aspectmode="data"),
            autosize=True,
            width=width,
            height=height,
            margin=dict(l=10, r=10, b=10, t=40 if name else 10, pad=4),
            paper_bgcolor="White",
            title=name))
        self.fig.update_layout(legend={'itemsizing': 'constant'})
        if bigfont: self.fig.update_layout(legend_font_size=16, title_font_size=20)
        self.shown_legendgroups = [] #https://stackoverflow.com/a/26940058/5122790

    def _transform(self, points, inverse=False):
        trafo_fn = self.back_trafo_fn if inverse else self.trafo_fn
        points = np.array([trafo_fn(point) for point in points])
        if self.swap_axes:
            points = self._swap_axes(points, self.swap_axes)
        return points

    def _swap_axes(self, points, swap_axes):
        swap_translate = {"x": 0, "y": 1, "z": 2}
        ind1, ind2 = swap_translate[swap_axes[0]], swap_translate[swap_axes[1]]
        tmp = points[:, ind1].copy()
        points[:, ind1] = points[:, ind2]
        points[:, ind2] = tmp
        return points

    def _get_meshgrid(self, plane, samples, margin):
        trafo, back_trafo = make_base_changer(plane)
        onto_plane = np.array([back_trafo([0, trafo(point)[1], trafo(point)[2]]) for point in samples])
        minx, miny, minz = onto_plane.min(axis=0)
        maxx, maxy, maxz = onto_plane.max(axis=0)
        xx, yy = make_meshgrid(minx=minx, miny=miny, maxx=maxx, maxy=maxy, margin=margin)
        return xx, yy, minz, maxz

    def _get_surface_tight(self, plane, samples, margin):
        xx, yy, minz, maxz = self._get_meshgrid(plane, samples, margin)
        xy_arr = np.vstack([xx.flatten(), yy.flatten()]).T
        col_fn = lambda x, y: 0 if minz - margin < plane.z(np.array([x,y])) < maxz + margin else 1
        z_fn = lambda x, y: min(max(plane.z(np.array([x, y])), minz - margin), maxz + margin)
        cols = np.array([col_fn(*xy) for xy in xy_arr])
        z_arr = np.array([z_fn(*xy) for xy in xy_arr])
        points = np.column_stack([xy_arr, z_arr])
        surface_form = lambda x: x.reshape(round(math.sqrt(x.shape[0])), -1)
        res = {k: surface_form(points[:, ind]) for ind, k in enumerate("xyz")}
        return {**res, **{"surfacecolor": surface_form(cols)}}

    def add_surface(self, plane, samples, labels=None, margin=0, swap_axes=None, opacity=0.9, color="blue", showlegend=False, name=None):
        xx, yy, zz, cols = [v for v in self._get_surface_tight(plane, samples, margin).values()]
        points = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T
        points = self._transform(points)
        if swap_axes:
            points = self._swap_axes(points, swap_axes)
        surface_form = lambda x: x.reshape(round(math.sqrt(x.shape[0])), -1)
        self.fig.add_trace(
            go.Surface(x=surface_form(points[:, 0]),
                       y=surface_form(points[:, 1]),
                       z=surface_form(points[:, 2]),
                       surfacecolor=cols,
                       cmin=0,
                       cmax=1,
                       colorscale=[[0, color], [0.999, color], [1, "rgba(0,0,0,0)"]],
                       opacity=opacity, showlegend=showlegend, showscale=False, name=name,
                       )
        )

    def add_surface_old(self, xx, yy, z_func, swap_axes=None, opacity=None, color=None, showlegend=False, name=None): #TODO merge with ^
        xy_arr = np.vstack([xx.flatten(), yy.flatten()]).T
        z_arr = np.array([z_func(*i) for i in xy_arr])
        points = np.column_stack([xy_arr, z_arr])
        points = self._transform(points)
        if swap_axes:
            points = self._swap_axes(points, swap_axes)
        surface_form = lambda x: x.reshape(round(math.sqrt(x.shape[0])), -1)
        kwargs = dict(colorscale=[[0, color], [1, color]]) if color is not None else {}
        self.fig.add_trace(
            go.Surface(x=surface_form(points[:, 0]), y=surface_form(points[:, 1]), z=surface_form(points[:, 2]),
                       opacity=opacity, showlegend=showlegend, showscale=False, name=name, **kwargs))


    def add_line(self, point1, point2, width=6, do_transform=True, name=None, **kwargs):
        if do_transform:
            point1 = self._transform(np.array([point1])).squeeze()
            point2 = self._transform(np.array([point2])).squeeze()
        self.fig.add_trace(
            go.Scatter3d(x=[point1[0], point2[0]],
                         y=[point1[1], point2[1]],
                         z=[point1[2], point2[2]],
                         marker=dict(size=1),
                         line=dict(width=width),
                         name=name,
                         **kwargs
                         )
        )

    def add_markers(self, points, color="black", size=2, name=None, custom_data=None, linelen_left=25, linelen_right=60, maxlen=500, **kwargs):
        points = np.array(points)
        if points.ndim == 1: points = np.array([points])
        points = self._transform(points)
        default_args = dict(mode="markers", x=points[:, 0], y=points[:, 1], z=points[:, 2],
                            marker={"color": color, "size": size, "line": {"width": 0}}, name=name)
        all_args = {**default_args, **kwargs}
        if all_args.get(name) and all_args.get("showlegend") is not None:
            all_args["showlegend"] = True
        #print({k:v for k,v in all_args.items() if k not in "xyz"})
        if custom_data:
            keys = [i for i in custom_data[0].keys() if i != "extra"]
            all_args["customdata"] = [list([v for k,v in i.items() if k != "extra"]) for i in custom_data]
            hovertemplate = "<br>".join([self.hovertemplate]+[f"{key}: %{{customdata[{i}]}}" for i, key in enumerate(keys)])
            if "extra" in custom_data[0].keys():
                extra_keys = [i for i in custom_data[0]["extra"].keys()]
                all_args["customdata"] = [list([v for k,v in i.items() if k.lower() != "extra"])+list(i["extra"].values()) for i in custom_data]
                hovertemplate = hovertemplate+"<extra>"+"<br>".join([f"{key}: %{{customdata[{i+len(keys)}]}}" for i, key in enumerate(extra_keys)])+"</extra>"
                all_args["customdata"] = [[("<br>" if n >= len(keys) else "")+"<br>".join(textwrap.wrap(textwrap.shorten(str(j), maxlen), linelen_left if n < len(keys) else linelen_right)) for n, j in enumerate(i)] for i in all_args["customdata"]]
        else:
            hovertemplate = self.hovertemplate
        trace = go.Scatter3d(**all_args)
        trace.update(hovertemplate=hovertemplate, textposition="top left")
        self.fig.add_trace(trace)


    def add_sample_projections(self, X, onto, n_samples=10, **kwargs):
        show_vecs = X[np.random.choice(X.shape[0], n_samples, replace=False), :]
        for point in show_vecs:
            proj = ortho_projection_affine(point, onto)
            self.add_line(point, proj, **kwargs)

    def add_quader(self, coords, name=None, color="#DC143C", opacity=0.6):
        self.fig.add_trace(
            go.Mesh3d(
                x=coords[0], y=coords[1], z=coords[2],
                i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                opacity=opacity,
                color=color,
                flatshading=True,
                name=name,
                showlegend=name not in self.shown_legendgroups,
                legendgroup=name,
            )
        )
        self.shown_legendgroups.append(name)


    def __enter__(self, *args):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if not exc_type:
            return self.fig

    def show(self):
        self.fig.show()

