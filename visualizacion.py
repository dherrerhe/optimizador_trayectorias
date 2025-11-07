# visualizacion.py
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

def quiver_2d(F_np, xlim=(-0.1,1.2), ylim=(-0.1,1.2), density=20):
    X, Y = np.meshgrid(np.linspace(*xlim, density), np.linspace(*ylim, density))
    grid = np.stack((X, Y), axis=-1)
    V = F_np(grid)
    Ux, Uy = V[...,0], V[...,1]
    fig = ff.create_quiver(X, Y, Ux, Uy, scale=0.2, arrow_scale=.4)
    fig.update_layout(width=600, height=600, xaxis_title="x", yaxis_title="y")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

def add_path(fig, P, name="trayectoria"):
    fig.add_trace(go.Scatter(x=P[:,0], y=P[:,1], mode="lines", name=name))
    return fig

def potencial_3d(potential_fn, xlim=(-1.5,1.5), ylim=(-1.5,1.5), density=60, title="Potencial"):
    X = np.linspace(*xlim, density)
    Y = np.linspace(*ylim, density)
    XX, YY = np.meshgrid(X, Y)
    ZZ = potential_fn(XX, YY)
    fig = go.Figure(data=[go.Surface(x=XX, y=YY, z=ZZ, opacity=0.9)])
    fig.update_layout(title=title, scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x,y)"),
                      width=700, height=500)
    return fig

def plot_W_vs_a(a_vals, W_vals, title="Trabajo vs parámetro a"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a_vals, y=W_vals, mode="lines+markers", name="W(a)"))
    fig.update_layout(title=title, xaxis_title="a", yaxis_title="Trabajo W(a)", width=700, height=450)
    return fig
