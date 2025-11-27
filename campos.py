import numpy as np
import sympy as sp

x, y = sp.symbols('x y', real=True)

def crear_campo(nombre, P_expr, Q_expr):
    """
    Crea una entrada para FIELDS a partir de P(x,y) y Q(x,y) simbólicos.
    Calcula, si es posible, el potencial para campos conservativos.
    Incluye validaciones numéricas y manejo seguro de errores.
    """
    F_sym = (P_expr, Q_expr)
    curl = sp.simplify(curl_2d(F_sym))
    es_conservativo = (curl == 0)

    # Versión NumPy de P y Q
    P_l = sp.lambdify((x, y), P_expr, "numpy")
    Q_l = sp.lambdify((x, y), Q_expr, "numpy")

    def F_np(xy: np.ndarray) -> np.ndarray:
        X = xy[..., 0]
        Y = xy[..., 1]
        return np.stack((P_l(X, Y), Q_l(X, Y)), axis=-1)

    # Validación: probar evaluaciones
    try:
        pv = P_l(0.0, 0.0)
        qv = Q_l(0.0, 0.0)
        _ = float(pv) + float(qv)  # fuerza conversión a float/ndarray
    except Exception as e:
        raise ValueError(f"Las funciones P o Q no se pueden evaluar numéricamente en (0,0): {e}")

    potencial_fn = None
    if es_conservativo:
        try:
            # Calcula el potencial de manera directa, sin simplify
            f_int_x = sp.integrate(P_expr, x)
            df_dy = sp.diff(f_int_x, y)
            resto = sp.simplify(Q_expr - df_dy)
            C_y = sp.integrate(resto, y)
            f_pot = f_int_x + C_y
            # Solo simplifica si es MUY largo
            if len(str(f_pot)) > 200:
                f_pot = sp.simplify(f_pot)
            potencial_fn = sp.lambdify((x, y), f_pot, "numpy")
            # Validación del potencial
            vpot = potencial_fn(0.0, 0.0)
            _ = float(vpot)
        except Exception as e:
            potencial_fn = None  # No hay potencial fiable/calculable

    return nombre, {
        "sym": F_sym,
        "np": F_np,
        "conservativo": es_conservativo,
        "curl": curl,
        "potencial": potencial_fn,
    }


def curl_2d(PQ_sym) -> sp.Expr:
    """
    Calcula el rotacional ("curl") en 2D para un campo dado
    como una tupla de expresiones simbólicas (P, Q).
    """
    P, Q = PQ_sym
    return sp.diff(Q, x) - sp.diff(P, y)

# Definimos los símbolos simbólicos para las variables x e y
x, y = sp.symbols('x y', real=True)

# Campo CONSERVATIVO F1 = ∇f, con f = x^2 + y^2
f1 = x ** 2 + y ** 2  # Potencial escalar para el campo conservativo
F1_sym = (sp.diff(f1, x), sp.diff(f1, y))  # Gradiente del potencial: (2x, 2y)

def campo_conservativo(xy: np.ndarray) -> np.ndarray:
    """
    Versión numérica de F1.
    Calcula el campo F1 = (2x, 2y) dada una entrada de puntos xy.
    """
    X = xy[..., 0]
    Y = xy[..., 1]
    return np.stack((2 * X, 2 * Y), axis=-1)

# Campo NO CONSERVATIVO F2 = (-y, x)
F2_sym = (-y, x)  # Campo definido simbólicamente como una tupla (P, Q)

def campo_rotacional(xy: np.ndarray) -> np.ndarray:
    """
    Versión numérica de F2.
    Calcula el campo F2 = (-y, x) dada una entrada de puntos xy.

    Parámetros
    ----------
    xy : np.ndarray
        Un arreglo de N puntos en R^2. Es decir, un array de forma (N, 2) donde 
        cada fila representa un punto (x, y). np.ndarray (abreviatura de "NumPy array") 
        es una estructura de datos fundamental de la librería NumPy que permite 
        almacenar y operar eficientemente sobre datos numéricos en arreglos multidimensionales.

    Retorna
    -------
    np.ndarray
        Un array de forma (N, 2) con los valores del campo F2 en cada punto.
    """
    X = xy[..., 0]
    Y = xy[..., 1]
    return np.stack((-Y, X), axis=-1)

# ----- Utilidades -----

def potencial_f1(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Potencial de F1 (solo existe para el campo conservativo).
    Devuelve X^2 + Y^2.
    """
    return X ** 2 + Y ** 2

# Rotacionales para los campos definidos anteriormente
CURL_F1 = sp.simplify(curl_2d(F1_sym))   # = 0 para campo conservativo
CURL_F2 = sp.simplify(curl_2d(F2_sym))   # = 2 para campo no conservativo

FIELDS = {
    # Diccionario con información sobre cada campo vectorial
    "Conservativo: F1(x,y) = (2x, 2y)": {
        "sym": F1_sym,
        "np": campo_conservativo,
        "conservativo": True,
        "curl": CURL_F1,
        "potencial": potencial_f1,
    },
    "No conservativo: F2(x,y) = (-y, x)": {
        "sym": F2_sym,
        "np": campo_rotacional,
        "conservativo": False,
        "curl": CURL_F2,
        "potencial": None,
    },
}