import streamlit as st
import numpy as np
import sympy as sp

from campos import FIELDS, crear_campo, x, y
from integrales import (
    calcular_trabajo, trayectoria_recta, velocidad_recta,
    trayectoria_parabolica, velocidad_parabolica,
    trayectoria_parametrica, velocidad_parametrica
)
from visualizacion import quiver_2d, add_path, potencial_3d, plot_W_vs_a

# -------------------------------------------------
# Configuración básica de la página de Streamlit
# -------------------------------------------------
st.set_page_config(page_title="Optimizador de Trayectorias", layout="wide")
st.title("🧭 Optimizador de Trayectorias — Conservativos vs No Conservativos")

# -------------------------------------------------
# SIDEBAR: selección de campo, editor de campos y parámetros
# -------------------------------------------------
st.sidebar.header("⚙ Selección del campo")

# Menú desplegable para seleccionar campos existentes
campo_nombre = st.sidebar.selectbox("Campo vectorial", list(FIELDS.keys()))
campo = FIELDS[campo_nombre]
F_np = campo["np"]

# Editor de campos personalizados en el sidebar
st.sidebar.subheader("➕ Crear un nuevo campo")

nombre_nuevo = st.sidebar.text_input("Nombre del campo", "Campo personalizado")

expr_P = st.sidebar.text_input("P(x,y) =", "2*x")# Input para la expresión simbólica de P(x,y), componente x del campo

expr_Q = st.sidebar.text_input("Q(x,y) =", "-y") # Input para la expresión simbólica de Q(x,y), componente y del campo

# Botón para que el usuario confirme la creación del nuevo campo
if st.sidebar.button("Añadir campo"):
    try:
        # Interpreta (parsea) de forma segura las expresiones de P y Q como expresiones de sympy,
        # permitiendo solo variables x, y, e y pi en el entorno de sympify (evita inseguridades)
        P_expr = sp.sympify(expr_P, {"x": x, "y": y, "e": sp.E, "pi": sp.pi})
        Q_expr = sp.sympify(expr_Q, {"x": x, "y": y, "e": sp.E, "pi": sp.pi})

        # Crea el nuevo campo utilizando la función de utilidades, incluyendo nombre descriptivo
        # Ejemplo de nombre: "MiCampo: (2*x, -y)"
        nombre_creado, campo_creado = crear_campo(
            f"{nombre_nuevo}: ({expr_P}, {expr_Q})",
            P_expr,
            Q_expr
        )
        FIELDS[nombre_creado] = campo_creado # Agrega el campo creado al diccionario global FIELDS para que esté disponible en el resto de la app
        st.sidebar.success(f"Campo '{nombre_creado}' añadido ") # Mensaje de éxito para el usuario en el sidebar

    except Exception as e:
        st.sidebar.error(f"Error al interpretar P y Q:\n{e}") # En caso de error de parsing o construcción, muestra mensaje de error amigable al usuario

#--------Sección para ingresar/manipular los puntos A y B (extremos de las trayectorias)--------
# (Esto permite al usuario definir el punto inicial y final para las curvas de integración)

st.sidebar.markdown("---")   # Línea divisoria en el sidebar
st.sidebar.write("**Puntos extremos** (A y B)")  # Título descriptivo en sidebar

# Crea inputs numéricos para los valores x e y de los puntos A y B (con valores por defecto y step)
Ax = st.sidebar.number_input("A_x", value=0.0, step=0.1)   # Componente x de A
Ay = st.sidebar.number_input("A_y", value=0.0, step=0.1)   # Componente y de A
Bx = st.sidebar.number_input("B_x", value=1.0, step=0.1)   # Componente x de B
By = st.sidebar.number_input("B_y", value=1.0, step=0.1)   # Componente y de B

# Construye los puntos A y B como arrays de numpy para usarlos en cálculos y trayectorias
A = np.array([Ax, Ay])
B = np.array([Bx, By])

# Sección para los parámetros de trayectoria 2
st.sidebar.markdown("---")
tray_sel = st.sidebar.selectbox(
    "Trayectoria 2",
    ["Curva parabólica (t, t^2)",
     "Familia cuadrática y=(1-a)t + a t^2"]
)
a = st.sidebar.slider("Parámetro a (solo familia)", -2.0, 2.0, 1.0, 0.1)

# Parámetros numéricos de integración y visualización
res = st.sidebar.slider("Resolución integración (n)", 500, 6000, 2000, 100)
dens = st.sidebar.slider("Densidad del campo (flechas)", 10, 30, 20, 1)

# Opción para explorar el óptimo W(a)
st.sidebar.markdown("---")
explorar_optimo = False
if (not campo["conservativo"]) and tray_sel.startswith("Familia"):
    explorar_optimo = st.sidebar.checkbox("Explorar W(a) en la familia", value=False)
else:
    st.sidebar.caption(
        "El barrido W(a) solo aplica para campos no conservativos y la familia cuadrática."
    )

# -------------------------------------------------
# CUERPO PRINCIPAL: información del campo, trabajos y gráficas
# -------------------------------------------------

# Expansor con detalles del campo seleccionado
with st.expander("Detalles del campo"):
    st.write(f"**Campo seleccionado:** {campo_nombre}")
    st.write(f"**Conservativo:** {campo['conservativo']}  |  **curl** = `{campo['curl']}`")
    if campo["conservativo"]:
        if campo["potencial"] is not None:
            st.write("Tiene potencial escalar (se muestra más abajo).")
        else:
            st.write("Es conservativo (curl = 0), pero no se pudo obtener el potencial automáticamente.")

# Definición de las trayectorias según la selección del usuario
r1, dr1 = trayectoria_recta(A, B), velocidad_recta(A, B)
if tray_sel.startswith("Curva"):
    r2, dr2 = trayectoria_parabolica(A, B), velocidad_parabolica(A, B)
else:
    r2, dr2 = trayectoria_parametrica(A, B, a), velocidad_parametrica(A, B, a)

# Cálculo de los trabajos sobre ambas trayectorias
W1 = calcular_trabajo(F_np, r1, dr1, n=res)
W2 = calcular_trabajo(F_np, r2, dr2, n=res)

# Visualización rápida de valores de trabajo en columnas
col1, col2 = st.columns(2)
with col1:
    # Muestra un cuadro métrico con el resultado del trabajo calculado a lo largo de la trayectoria recta entre A y B
    st.metric("Trabajo en recta", f"{W1:.6f}")
with col2:
    # Muestra un cuadro métrico con el resultado del trabajo calculado a lo largo de la segunda trayectoria
    st.metric("Trabajo trayectoria 2", f"{W2:.6f}")

# Generación de los puntos a graficar para ambas trayectorias
t_plot = np.linspace(0, 1, 500)
P1 = r1(t_plot)
P2 = r2(t_plot)

# Ajuste automático de los límites de la gráfica según las trayectorias
# Calcula los límites mínimos y máximos en x y en y, considerando ambas trayectorias (P1 y P2)
x_min = min(P1[:, 0].min(), P2[:, 0].min())  # mínimo x entre ambas trayectorias
x_max = max(P1[:, 0].max(), P2[:, 0].max())  # máximo x entre ambas trayectorias
y_min = min(P1[:, 1].min(), P2[:, 1].min())  # mínimo y entre ambas trayectorias
y_max = max(P1[:, 1].max(), P2[:, 1].max())  # máximo y entre ambas trayectorias

# Calcula la diferencia (rango) en x y en y
dx = x_max - x_min
dy = y_max - y_min

# Padding: si el rango es 0, usa 1.0; si no, usa 20% del rango
pad_x = 0.2 * dx if dx > 0 else 1.0
pad_y = 0.2 * dy if dy > 0 else 1.0

# Define los límites finales con padding
xlim = (x_min - pad_x, x_max + pad_x)
ylim = (y_min - pad_y, y_max + pad_y)

# Gráfico del campo vectorial junto a ambas trayectorias
fig = quiver_2d(F_np, xlim=xlim, ylim=ylim, density=dens)
fig = add_path(fig, P1, "Recta A→B")
fig = add_path(fig, P2, "Trayectoria 2")

st.plotly_chart(fig, use_container_width=True, key="campo_2d")

# Si es un campo conservativo y existe potencial, grafica la superficie
if campo["conservativo"] and campo["potencial"] is not None:
    st.subheader("Superficie del potencial (solo campos conservativos)")
    f3d = potencial_3d(campo["potencial"], title="f(x,y)")
    st.plotly_chart(f3d, use_container_width=True, key="potencial_3d")

# Barrido del parámetro a en la familia de trayectorias, si corresponde
if explorar_optimo and tray_sel.startswith("Familia"):
    st.subheader("Barrido del parámetro a")
    a_vals = np.linspace(-2.0, 2.0, 81)
    W_vals = []
    for av in a_vals:
        rA, dA = trayectoria_parametrica(A, B, av), velocidad_parametrica(A, B, av)
        W_vals.append(calcular_trabajo(F_np, rA, dA, n=res))
    st.plotly_chart(
        plot_W_vs_a(a_vals, W_vals),
        use_container_width=True,
        key="barrido_Wa"
    )

# Mensaje informativo para el usuario sobre la naturaleza del trabajo en diferentes campos
st.info(
    "Tip: en campos conservativos el trabajo solo depende de A y B; "
    "en campos no conservativos, depende de la forma de la trayectoria."
)
