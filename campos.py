import numpy as np
import sympy as sp

# ----- Definiciones simbólicas -----

# Definimos los símbolos simbólicos para las variables x e y
x, y = sp.symbols('x y', real=True)

# Campo CONSERVATIVO F1 = ∇f, con f = x^2 + y^2
f1 = x ** 2 + y ** 2  # Potencial escalar para el campo conservativo
F1_sym = (sp.diff(f1, x), sp.diff(f1, y))  # Gradiente del potencial: (2x, 2y)

def F1_np(xy: np.ndarray) -> np.ndarray:
    """
    Versión numérica de F1.
    Calcula el campo F1 = (2x, 2y) dada una entrada de puntos xy.
    """
    X = xy[..., 0]
    Y = xy[..., 1]
    return np.stack((2 * X, 2 * Y), axis=-1)

# Campo NO CONSERVATIVO F2 = (-y, x)
F2_sym = (-y, x)  # Campo definido simbólicamente como una tupla (P, Q)

def F2_np(xy: np.ndarray) -> np.ndarray:
    """
    Versión numérica de F2.
    Calcula el campo F2 = (-y, x) dada una entrada de puntos xy.
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

def curl_2d(PQ_sym) -> sp.Expr:
    """
    Calcula el rotacional ("curl") en 2D para un campo dado
    como una tupla de expresiones simbólicas (P, Q).
    """
    P, Q = PQ_sym
    return sp.diff(Q, x) - sp.diff(P, y)

# Rotacionales para los campos definidos anteriormente
CURL_F1 = sp.simplify(curl_2d(F1_sym))   # = 0 para campo conservativo
CURL_F2 = sp.simplify(curl_2d(F2_sym))   # = 2 para campo no conservativo

FIELDS = {
    # Diccionario con información sobre cada campo vectorial
    "Conservativo: F1(x,y) = (2x, 2y)": {
        "sym": F1_sym,
        "np": F1_np,
        "conservativo": True,
        "curl": CURL_F1,
        "potencial": potencial_f1,
    },
    "No conservativo: F2(x,y) = (-y, x)": {
        "sym": F2_sym,
        "np": F2_np,
        "conservativo": False,
        "curl": CURL_F2,
        "potencial": None,
    },
}
