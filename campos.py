import numpy as np
import sympy as sp

x, y = sp.symbols('x y', real=True)

def crear_campo(nombre, P_expr, Q_expr):
    """
    Crea la entrada para FIELDS a partir de P(x,y) y Q(x,y) simbólicos.
    Si el campo es conservativo (curl = 0), intenta hallar el potencial f(x,y).
    """
    F_sym = (P_expr, Q_expr)
    curl = sp.simplify(curl_2d(F_sym))
    es_conservativo = (curl == 0)

    # Genera funciones NumPy a partir de las expresiones simbólicas
    P_l = sp.lambdify((x, y), P_expr, "numpy")
    Q_l = sp.lambdify((x, y), Q_expr, "numpy")

    def F_np(xy: np.ndarray) -> np.ndarray:
        # Extrae coordenadas X e Y del arreglo de puntos
        X = xy[..., 0]
        Y = xy[..., 1]
        # Calcula el vector para cada punto y lo apila en el eje -1
        return np.stack((P_l(X, Y), Q_l(X, Y)), axis=-1)

    potencial_fn = None  # Inicializa función de potencial como None

    if es_conservativo:
        # Paso 1: integrar P respecto de x para obtener la primitiva parcial
        f_int_x = sp.integrate(P_expr, x)  # f(x, y) + C(y)

        # Paso 2: derivar esa primitiva respecto de y
        df_dy = sp.diff(f_int_x, y)

        # Paso 3: C'(y) = Q(x, y) - df_dy
        resto = sp.simplify(Q_expr - df_dy)

        # Paso 4: integra el residuo respecto de y para hallar C(y)
        # (si depende solo de y, da el potencial correcto)
        C_y = sp.integrate(resto, y)

        # Suma ambas partes para obtener el potencial completo
        f_pot = f_int_x + C_y
        f_pot_simpl = sp.simplify(f_pot)

        # Convierte el potencial simbólico a función NumPy
        potencial_fn = sp.lambdify((x, y), f_pot_simpl, "numpy")

    # Devuelve un diccionario con toda la información relevante del campo
    return nombre, {
        "sym": F_sym,                # Tupla con las funciones simbólicas
        "np": F_np,                  # Función NumPy para evaluar el campo
        "conservativo": es_conservativo,  # Booleano: True si es conservativo
        "curl": curl,                # Valor del rotacional (curl)
        "potencial": potencial_fn,   # Función NumPy del potencial si existe
    }

# ----- Definiciones simbólicas -----

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
    # Extrae las coordenadas x e y de cada punto
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
