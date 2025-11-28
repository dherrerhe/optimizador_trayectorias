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
expr_P = st.sidebar.text_input("P(x,y) =", "2*x")
expr_Q = st.sidebar.text_input("Q(x,y) =", "-y")

# Si se presiona el botón para añadir campo, intenta crear el campo personalizado
if st.sidebar.button("Añadir campo"):
    try:
        P_expr = sp.sympify(expr_P, {"x": x, "y": y, "e": sp.E, "pi": sp.pi})
        Q_expr = sp.sympify(expr_Q, {"x": x, "y": y, "e": sp.E, "pi": sp.pi})
        nombre_creado, campo_creado = crear_campo(
            f"{nombre_nuevo}: ({expr_P}, {expr_Q})",
            P_expr,
            Q_expr
        )
        FIELDS[nombre_creado] = campo_creado
        st.sidebar.success(f"Campo '{nombre_creado}' añadido ")
    except Exception as e:
        st.sidebar.error(f"Error al interpretar P y Q:\n{e}")

st.sidebar.markdown("---")
st.sidebar.write("**Puntos extremos** (A y B)")
Ax = st.sidebar.number_input("A_x", value=0.0, step=0.1)
Ay = st.sidebar.number_input("A_y", value=0.0, step=0.1)
Bx = st.sidebar.number_input("B_x", value=1.0, step=0.1)
By = st.sidebar.number_input("B_y", value=1.0, step=0.1)
A = np.array([Ax, Ay])
B = np.array([Bx, By])

st.sidebar.markdown("---")

tray_sel = st.sidebar.selectbox(
    "Trayectoria 2",
    [
        "Curva parabólica (t, t^2)",
        "Familia cuadrática y=(1-a)t + a t^2"
    ]
)

a = st.sidebar.slider("Parámetro a (solo familia)", -2.0, 2.0, 1.0, 0.1)
res = st.sidebar.slider("Resolución integración (n)", 500, 6000, 2000, 100)
dens = st.sidebar.slider("Densidad del campo (flechas)", 10, 30, 20, 1)

st.sidebar.markdown("---")

search_min = False
explorar_optimo = False

# Si el campo no es conservativo y se selecciona la familia cuadrática,
# se habilitan las opciones de exploración y búsqueda óptima
if (not campo["conservativo"]) and tray_sel.startswith("Familia"):
    col_opt1, col_opt2 = st.sidebar.columns(2)
    explorar_optimo = col_opt1.checkbox("Explorar W(a)", value=False)
    search_min = col_opt2.button("search")
else:
    st.sidebar.caption(
        # Solo tiene sentido explorar W(a) para campos no conservativos y la familia cuadrática
        "El barrido W(a) solo aplica para campos no conservativos y la familia cuadrática."
    )

# -------------------------------------------------
# CUERPO PRINCIPAL: información del campo, trabajos y gráficas
# -------------------------------------------------
with st.expander("Detalles del campo"):
    st.write(f"**Campo seleccionado:** {campo_nombre}")
    st.write(
        f"**Conservativo:** {campo['conservativo']}  |  **curl** = `{campo['curl']}`"
    )
    if campo["conservativo"]:
        if campo["potencial"] is not None:
            st.write("Tiene potencial escalar (se muestra más abajo).")
        else:
            st.write(
                "Es conservativo (curl = 0), pero no se pudo obtener el potencial automáticamente."
            )

# Generación de las trayectorias y sus derivadas según la selección
r1, dr1 = trayectoria_recta(A, B), velocidad_recta(A, B)
if tray_sel.startswith("Curva"):
    r2, dr2 = trayectoria_parabolica(A, B), velocidad_parabolica(A, B)
else:
    r2, dr2 = trayectoria_parametrica(A, B, a), velocidad_parametrica(A, B, a)

# Cálculo de los trabajos en ambas trayectorias
W1 = calcular_trabajo(F_np, r1, dr1, n=res)
W2 = calcular_trabajo(F_np, r2, dr2, n=res)

# Métricas mostradas al usuario sobre el trabajo en cada trayectoria
col1, col2 = st.columns(2)
with col1:
    st.metric("Trabajo en recta", f"{W1:.6f}")
with col2:
    st.metric("Trabajo trayectoria 2", f"{W2:.6f}")

# Preparación de datos para graficar las trayectorias, determinando límites del gráfico
t_plot = np.linspace(0, 1, 500)
P1 = r1(t_plot)
P2 = r2(t_plot)
x_min = min(P1[:, 0].min(), P2[:, 0].min())
x_max = max(P1[:, 0].max(), P2[:, 0].max())
y_min = min(P1[:, 1].min(), P2[:, 1].min())
y_max = max(P1[:, 1].max(), P2[:, 1].max())
dx = x_max - x_min
dy = y_max - y_min
pad_x = 0.2 * dx if dx > 0 else 1.0
pad_y = 0.2 * dy if dy > 0 else 1.0
xlim = (x_min - pad_x, x_max + pad_x)
ylim = (y_min - pad_y, y_max + pad_y)

# === ÚNICA GRÁFICA: General o con Óptima incluida ===
fig = quiver_2d(F_np, xlim=xlim, ylim=ylim, density=dens)
fig = add_path(fig, P1, "Recta A→B")
fig = add_path(fig, P2, "Trayectoria 2")

# Búsqueda de parámetro óptimo a para familia cuadrática en campos no conservativos
# Si se solicita buscar el mínimo y se cumple que la trayectoria es de la familia cuadrática
# y el campo no es conservativo, se procede a buscar el parámetro óptimo 'a' que minimiza el trabajo.
if search_min and tray_sel.startswith("Familia") and (not campo["conservativo"]):
    # Genera 201 valores de 'a' entre -2 y 2 (barrido fino)
    a_vals = np.linspace(-2.0, 2.0, 201)
    W_vals = []  # Lista para ir guardando los trabajos calculados para cada 'a'
    for av in a_vals:
        # Para cada valor de 'a', genera la trayectoria y su velocidad asociada
        rA, dA = trayectoria_parametrica(A, B, av), velocidad_parametrica(A, B, av)
        # Calcula el trabajo realizado por el campo en dicha trayectoria
        W_vals.append(calcular_trabajo(F_np, rA, dA, n=res))
    W_vals = np.array(W_vals)  # Convierte la lista de trabajos a un array de numpy
    idx_min = np.argmin(W_vals)  # Encuentra el índice del valor mínimo de trabajo
    a_opt = a_vals[idx_min]      # El valor de 'a' óptimo corresponde al mínimo encontrado
    W_opt = W_vals[idx_min]      # El trabajo mínimo correspondiente
    # Muestra la información sobre la trayectoria óptima encontrada
    st.info(
        f"La trayectoria de mínimo trabajo tiene a = {a_opt:.4f} "
        f"(Trabajo mínimo = {W_opt:.6f})"
    )
    # Traza la trayectoria óptima encontrada en la figura y la añade como "Óptima"
    P_opt = trayectoria_parametrica(A, B, a_opt)(t_plot)
    fig = add_path(fig, P_opt, f"Óptima a={a_opt:.3f}")

# Muestra el gráfico principal de Streamlit
st.plotly_chart(fig, use_container_width=True)

# Si el campo es conservativo y tiene potencial escalar, grafica el potencial 3D
if campo["conservativo"] and campo["potencial"] is not None:
    st.subheader("Superficie del potencial (solo campos conservativos)")
    f3d = potencial_3d(campo["potencial"], title="f(x,y)")
    st.plotly_chart(f3d, use_container_width=True, key="potencial_3d")

# Si se selecciona explorar la variación de trabajo con a, grafica W(a)
# Si la opción de explorar el óptimo está habilitada y la trayectoria seleccionada es de la familia cuadrática
if explorar_optimo and tray_sel.startswith("Familia"):
    st.subheader("Barrido del parámetro a")  # Título descriptivo en la app
    a_vals = np.linspace(-2.0, 2.0, 81)  # Genera un rango de valores para el parámetro a
    W_vals = []  # Aquí se almacenarán los trabajos calculados para cada a
    # Para cada valor de a calcula la trayectoria y su derivada
    for av in a_vals:
        rA, dA = trayectoria_parametrica(A, B, av), velocidad_parametrica(A, B, av)
        # Calcula el trabajo para esa trayectoria y lo guarda
        W_vals.append(calcular_trabajo(F_np, rA, dA, n=res))
    # Grafica el trabajo en función de a en Streamlit usando Plotly
    st.plotly_chart(
        plot_W_vs_a(a_vals, W_vals),
        use_container_width=True,
        key="barrido_Wa"
    )

# Mensaje aclaratorio final sobre el significado del trabajo en campos conservativos vs no conservativos
st.info(
    "Tip: en campos conservativos el trabajo solo depende de A y B; "
    "en campos no conservativos, depende de la forma de la trayectoria."
)