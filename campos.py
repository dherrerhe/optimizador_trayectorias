import numpy as np
import sympy as sp

# Definición de símbolos simbólicos para x e y (real=True por default)
x, y = sp.symbols('x y', real=True)


def crear_campo(nombre, P_expr, Q_expr):
    """
    Crea una entrada para FIELDS a partir de P(x,y) y Q(x,y) simbólicos.
    Calcula, si es posible, el potencial para campos conservativos.
    Incluye validaciones numéricas y manejo seguro de errores.
    """
    F_sym = (P_expr, Q_expr)  # Tupla simbólica para el campo
    curl = sp.simplify(curl_2d(F_sym))
    es_conservativo = (curl == 0)  # Chequea si el campo es conservativo

    # Versiones NumPy de las funciones P(x, y) y Q(x, y)
    P_l = sp.lambdify((x, y), P_expr, "numpy")
    Q_l = sp.lambdify((x, y), Q_expr, "numpy")

    def F_np(xy: np.ndarray) -> np.ndarray:
        # Función vectorizada que evalúa el campo en un array de puntos [N, 2]
        X = xy[..., 0]
        Y = xy[..., 1]
        return np.stack((P_l(X, Y), Q_l(X, Y)), axis=-1)

    # Validación: intenta evaluar P y Q en el punto (0, 0)
    try:
        pv = P_l(0.0, 0.0)
        qv = Q_l(0.0, 0.0)
        _ = float(pv) + float(qv)  # Fuerza conversión a float/ndarray
    except Exception as e:
        raise ValueError(
            f"Las funciones P o Q no se pueden evaluar numéricamente en (0,0): {e}"
        )

    potencial_fn = None
    if es_conservativo:
        # Si es conservativo, intenta calcular el potencial escalar
        try:
            # Calcula el potencial por integración simbólica de P respecto de x
            f_int_x = sp.integrate(P_expr, x)
            # Calcula derivada parcial respecto de y para ajustar potencial
            df_dy = sp.diff(f_int_x, y)
            # Resto para ajustar C(y) por posibles términos faltantes de Q
            resto = sp.simplify(Q_expr - df_dy)
            # Integra el resto respecto de y
            C_y = sp.integrate(resto, y)
            # Suma de ambas partes
            f_pot = f_int_x + C_y
            # Si el potencial es muy complejo, simplifica la expresión
            if len(str(f_pot)) > 200:
                f_pot = sp.simplify(f_pot)
            # Compila función NumPy del potencial
            potencial_fn = sp.lambdify((x, y), f_pot, "numpy")
            # Chequea que la función potencial se pueda evaluar
            vpot = potencial_fn(0.0, 0.0)
            _ = float(vpot)
        except Exception as e:
            potencial_fn = None  # No hay potencial fiable/calculable

    # Devuelve un tuple (nombre, diccionario de propiedades)
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
    # En R^2 el rotacional es ∂Q/∂x - ∂P/∂y
    return sp.diff(Q, x) - sp.diff(P, y)


# Definimos nuevamente los símbolos por claridad (según PEP 8,
# pero también pueden trasladarse a una única definición global).
x, y = sp.symbols('x y', real=True)

# Campo CONSERVATIVO: F1 = ∇f, con f = x^2 + y^2
f1 = x ** 2 + y ** 2  # Potencial escalar para el campo conservativo
F1_sym = (
    sp.diff(f1, x),  # 2x
    sp.diff(f1, y)   # 2y
)  # El gradiente del potencial: (2x, 2y)


def campo_conservativo(xy: np.ndarray) -> np.ndarray:
    """
    Versión numérica de F1.
    Calcula el campo F1 = (2x, 2y) dada una entrada de puntos xy.
    """
    X = xy[..., 0]
    Y = xy[..., 1]
    # Calcula (2x, 2y) para cada punto
    return np.stack((2 * X, 2 * Y), axis=-1)


# Campo NO CONSERVATIVO: F2 = (-y, x)
F2_sym = (-y, x)  # Campo definido simbólicamente como una tupla (P, Q)


def campo_rotacional(xy: np.ndarray) -> np.ndarray:
    """
    Versión numérica de F2.
    Calcula el campo F2 = (-y, x) dada una entrada de puntos xy.

    Parámetros
    ----------
    xy : np.ndarray
        Un arreglo de N puntos en R^2 (array de shape (N, 2)).
        Cada fila representa un punto (x, y).

    Retorna
    -------
    np.ndarray
        Un array de shape (N, 2) con los valores del campo F2 en cada punto.
    """
    X = xy[..., 0]
    Y = xy[..., 1]
    # Devuelve el vector (-y, x) para cada punto del array de entrada
    return np.stack((-Y, X), axis=-1)


# ----- Utilidades -----


def potencial_f1(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Potencial de F1 (solo existe para el campo conservativo).
    Devuelve X^2 + Y^2.
    """
    return X ** 2 + Y ** 2


# Rotacionales para los campos definidos anteriormente (simbolicamente)
CURL_F1 = sp.simplify(curl_2d(F1_sym))  # = 0 para campo conservativo
CURL_F2 = sp.simplify(curl_2d(F2_sym))  # = 2 para campo no conservativo

# Diccionario con información sobre cada campo vectorial
FIELDS = {
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