# visualizacion.py
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

def quiver_2d(F_np, xlim=(-0.1, 1.2), ylim=(-0.1, 1.2), density=20):
    """
    Dibuja un campo de flechas (quiver) 2D para un campo vectorial dado.

    Parámetros:
        F_np : función que recibe un grid de puntos y devuelve el campo vectorial evaluado.
        xlim : tupla, límites del eje x.
        ylim : tupla, límites del eje y.
        density : int, número de flechas por dirección.

    Retorna:
        fig (plotly figure): figura con el quiver plot.
    """
    # Crear una malla de puntos en el dominio especificado
    X, Y = np.meshgrid(np.linspace(*xlim, density), np.linspace(*ylim, density))
    grid = np.stack((X, Y), axis=-1)
    # Calcular los vectores del campo en cada punto del grid
    V = F_np(grid)
    Ux, Uy = V[..., 0], V[..., 1]
    # Crear figura quiver usando plotly.figure_factory
    fig = ff.create_quiver(X, Y, Ux, Uy, scale=0.2, arrow_scale=.4)
    fig.update_layout(width=600, height=600, xaxis_title="x", yaxis_title="y")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def add_path(fig, P, name="trayectoria"):
    """
    Agrega una trayectoria (o camino) a una figura de plotly.

    Parámetros:
        fig : plotly figure, figura existente a la que agregar la trayectoria.
        P : np.ndarray, arreglo de Nx2 con las coordenadas del camino.
        name : str, nombre de la trayectoria para la leyenda.

    Retorna:
        fig (plotly figure): la figura modificada.
    """
    if "Recta" in name:
        color = "orange"
    elif "Trayectoria 2" in name or "Trayectoria" in name:
        color = "red"
    else:
        # color por defecto si se usa otro nombre
        color = "red"

    # Agrega a la figura la trayectoria especificada por los puntos P como una línea,
    # usando un color y ancho de línea determinados según el nombre de la trayectoria.
    fig.add_trace(
        go.Scatter(
            x=P[:, 0],      # Coordenada x de la trayectoria
            y=P[:, 1],      # Coordenada y de la trayectoria
            mode="lines",   # Dibuja solo la línea (no puntos)
            name=name,      # Nombre para la leyenda
            line=dict(color=color, width=3)  # Estilo de línea
        )
    )
    return fig


def potencial_3d(potential_fn, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
                 density=60, title="Potencial"):
    """
    Dibuja la superficie 3D del potencial escalar.

    Parámetros:
        potential_fn : función que devuelve el potencial f(x, y) evaluada sobre arrays.
        xlim : tupla, límites de x.
        ylim : tupla, límites de y.
        density : int, cantidad de puntos por eje.
        title : str, título de la figura.

    Retorna:
        fig (plotly figure): figura con la superficie 3D.
    """
    # Crear malla de puntos para evaluar el potencial
    X = np.linspace(*xlim, density)
    Y = np.linspace(*ylim, density)
    XX, YY = np.meshgrid(X, Y)
    ZZ = potential_fn(XX, YY)
    # Crear figura 3D usando plotly
    fig = go.Figure(data=[go.Surface(x=XX, y=YY, z=ZZ, opacity=0.9)])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x,y)"
        ),
        width=700,
        height=500
    )
    return fig


def plot_W_vs_a(a_vals, W_vals, title="Trabajo vs parámetro a"):
    """
    Grafica el trabajo W en función del parámetro 'a'.

    Parámetros:
        a_vals : array de valores de a.
        W_vals : array de valores de trabajo W(a).
        title : str, título de la gráfica.

    Retorna:
        fig (plotly figure): figura con la curva W(a).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=a_vals,
            y=W_vals,
            mode="lines+markers",
            name="W(a)"
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="a",
        yaxis_title="Trabajo W(a)",
        width=700,
        height=450
    )
    return fig
