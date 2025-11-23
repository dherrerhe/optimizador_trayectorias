import numpy as np
from typing import Callable

def calcular_trabajo(F_np: Callable, r: Callable, dr_dt: Callable,
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

def trayectoria_recta(A, B):
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

def velocidad_recta(A, B):
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

def trayectoria_parabolica(A, B):
    """
    Trayectoria curva que une A y B.
    x(t) cambia linealmente de Ax a Bx,
    y(t) cambia de Ay a By con forma parabólica.
    """
    Ax, Ay = A
    Bx, By = B

    def r(t):
        t = np.asarray(t)
        x = Ax + (Bx - Ax) * t          # interpolación lineal en x
        y = Ay + (By - Ay) * (t**2)     # parábola reescalada en y
        return np.column_stack((x, y))
    return r

def velocidad_parabolica(A, B):
    """
    Calcula la derivada (velocidad) para una trayectoria parabólica entre A y B.

    Parámetros
    ----------
    A : iterable de float
        Punto inicial (x, y).
    B : iterable de float
        Punto final (x, y).

    Retorna
    -------
    function
        Función dr/dt para la trayectoria parabólica.
    """
    Ax, Ay = A  # Extraer coordenadas iniciales
    Bx, By = B  # Extraer coordenadas finales

    def dr(t):
        # Convertimos t a un array de NumPy
        t = np.asarray(t)
        # La derivada de x(t) respecto a t es constante (Bx - Ax)
        dx = np.full_like(t, Bx - Ax)   # dx/dt de x(t) = Ax + (Bx-Ax)t
        # La derivada de y(t) respecto a t es 2*(By-Ay)*t (de y(t) = Ay + (By-Ay)t^2)
        dy = (By - Ay) * 2 * t          # dy/dt de y(t) = Ay + (By-Ay)t^2
        # Combinamos las derivadas en un arreglo columna
        return np.column_stack((dx, dy))
    return dr

def trayectoria_parametrica(A, B, a: float):
    """
    Familia de trayectorias entre A y B.
    x(t) lineal, y(t) = Ay + (By-Ay)*[(1-a)t + a t^2]
    """
    # Extrae las coordenadas iniciales y finales de los puntos A y B
    Ax, Ay = A
    Bx, By = B

    def r(t):
        # Convierte t en un array de NumPy, por si es escalar o vector
        t = np.asarray(t)
        # x(t) varía linealmente entre Ax y Bx
        x = Ax + (Bx - Ax) * t
        # Calcula la base cuadrática para y(t)
        base = (1 - a) * t + a * (t ** 2)
        # y(t) es una combinación lineal y cuadrática entre Ay y By
        y = Ay + (By - Ay) * base
        # Combina x e y en un arreglo de dos columnas (x, y)
        return np.column_stack((x, y))
    return r


def velocidad_parametrica(A, B, a: float):
    """Calcula la derivada (velocidad) de una trayectoria paramétrica cuadrática entre A y B.

    Args:
        A (np.ndarray): Punto inicial (Ax, Ay)
        B (np.ndarray): Punto final (Bx, By)
        a (float): Parámetro cuadrático

    Returns:
        function: función dr(t) que devuelve la velocidad en cada t
    """
    # Extrae las coordenadas iniciales y finales de los puntos A y B
    Ax, Ay = A
    Bx, By = B

    def dr(t):
        # Convierte t en un array de NumPy, por si llega escalar o vector
        t = np.asarray(t)
        # Derivada de x(t) respecto a t: siempre es (Bx - Ax), constante
        dx = np.full_like(t, Bx - Ax)
        # Derivada de [(1-a)t + a t^2] respecto a t: (1-a) + 2a t
        dbase_dt = (1 - a) + 2 * a * t
        # Derivada total respecto a t de y(t)
        dy = (By - Ay) * dbase_dt
        # Combina las derivadas en un solo arreglo de dos columnas (dx, dy)
        return np.column_stack((dx, dy))
    return dr