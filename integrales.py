import numpy as np
from typing import Callable

def integral_de_linea(F_np: Callable, r: Callable, dr_dt: Callable,
                      t0: float = 0.0, t1: float = 1.0, n: int = 2000) -> float:
    """
    Calcula W = ∫ F(r(t)) · r'(t) dt por regla del trapecio.

    Parámetros
    ----------
    F_np : Callable
        Función del campo vectorial (recibe array (N,2), devuelve array (N,2)).
    r : Callable
        Función de trayectoria r(t), devuelve array (N,2).
        Función de trayectoria r(t), devuelve array (N,2).
    dr_dt : Callable
        Derivada de la trayectoria, r'(t), devuelve array (N,2).
    t0 : float, opcional
        Valor inicial del parámetro t (default: 0.0).
    t1 : float, opcional
        Valor final del parámetro t (default: 1.0).
    n : int, opcional
        Número de puntos para la integración (default: 2000).

    Retorna
    -------
    float
        Aproximación numérica de la integral de línea.
    """
    t = np.linspace(t0, t1, n)           # Crea valores equiespaciados de t
    P = r(t)                             # Calcula los puntos de la trayectoria
    T = dr_dt(t)                         # Calcula los vectores tangentes a la trayectoria
    F = F_np(P)                          # Evalúa el campo en los puntos de la trayectoria
    integrando = np.sum(F * T, axis=1)   # Producto punto F·r' para cada t
    return float(np.trapz(integrando, t))  # Integra usando la regla del trapecio

# ---- Trayectorias estándar ----

def r_recta(A, B):
    """
    Retorna una función r(t) para la recta entre los puntos A y B.

    Parámetros
    ----------
    A : iterable de float
        Punto inicial (x, y).
    B : iterable de float
        Punto final (x, y).

    Retorna
    -------
    function
        Función r(t) que devuelve (x, y) para cada t.
    """
    def r(t):
        t = np.asarray(t)
        return np.column_stack((A[0] + (B[0] - A[0]) * t,
                                A[1] + (B[1] - A[1]) * t))
    return r

def dr_recta_dt(A, B):
    """
    Retorna una función dr/dt constante para una recta de A a B.

    Parámetros
    ----------
    A : iterable de float
        Punto inicial (x, y).
    B : iterable de float
        Punto final (x, y).

    Retorna
    -------
    function
        Función dr/dt para la recta, constante en t.
    """
    vx, vy = (B[0] - A[0]), (B[1] - A[1])
    def dr(t):
        t = np.asarray(t)
        return np.column_stack((np.full_like(t, vx), np.full_like(t, vy)))
    return dr

def r_parabola(A, B):
    """
    Retorna una función r(t) para una parábola canónica de (0,0) a (1,1).

    Parámetros
    ----------
    A : iterable de float
        Ignorado: la función asume A=(0,0).
    B : iterable de float
        Ignorado: la función asume B=(1,1).

    Retorna
    -------
    function
        Función r(t) = (t, t**2).
    """
    # Para simplicidad del proyecto fijamos A=(0,0), B=(1,1); si no, usar r_familia con 'a'.
    def r(t):
        t = np.asarray(t)
        return np.column_stack((t, t**2))
    return r

def dr_parabola_dt(A, B):
    """
    Retorna una función dr/dt para la parábola canónica (t, t**2).

    Parámetros
    ----------
    A : iterable de float
        Ignorado.
    B : iterable de float
        Ignorado.

    Retorna
    -------
    function
        Función dr/dt = (1, 2t).
    """
    def dr(t):
        t = np.asarray(t)
        return np.column_stack((np.ones_like(t), 2 * t))
    return dr

def r_familia(A, B, a: float):
    """
    Retorna una función r(t) para una familia cuadrática con parámetro 'a' de (0,0) a (1,1).

    Parámetros
    ----------
    A : iterable de float
        Ignorado: función asume A=(0,0).
    B : iterable de float
        Ignorado: función asume B=(1,1).
    a : float
        Parámetro de la familia cuadrática.

    Retorna
    -------
    function
        Función r(t) = (t, (1-a)t + a t**2).
    """
    def r(t):
        t = np.asarray(t)
        return np.column_stack((t, (1 - a) * t + a * (t ** 2)))
    return r

def dr_familia_dt(A, B, a: float):
    """
    Retorna función dr/dt para la familia cuadrática con 'a'.

    Parámetros
    ----------
    A : iterable de float
        Ignorado.
    B : iterable de float
        Ignorado.
    a : float
        Parámetro de la familia cuadrática.

    Retorna
    -------
    function
        Función dr/dt = (1, (1-a) + 2a t).
    """
    def dr(t):
        t = np.asarray(t)
        return np.column_stack((np.ones_like(t), (1 - a) + 2 * a * t))
    return dr
