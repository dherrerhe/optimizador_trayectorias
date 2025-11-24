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
# Configuración básica
# -------------------------------------------------
st.set_page_config(page_title="Optimizador de Trayectorias", layout="wide")
st.title("🧭 Optimizador de Trayectorias — Conservativos vs No Conservativos")

# -------------------------------------------------
# SIDEBAR: selección de campo + editor + parámetros
# -------------------------------------------------
st.sidebar.header("⚙ Selección del campo")

# Selector principal de campos existentes
campo_nombre = st.sidebar.selectbox("Campo vectorial", list(FIELDS.keys()))
campo = FIELDS[campo_nombre]
F_np = campo["np"]

# -------- Editor de campos dentro del sidebar --------
st.sidebar.subheader("➕ Crear un nuevo campo")

nombre_nuevo = st.sidebar.text_input("Nombre del campo", "Campo personalizado")
expr_P = st.sidebar.text_input("P(x,y) =", "2*x")
expr_Q = st.sidebar.text_input("Q(x,y) =", "-y")

if st.sidebar.button("Añadir campo"):
    try:
        P_expr = sp.sympify(expr_P, {"x": x, "y": y})
        Q_expr = sp.sympify(expr_Q, {"x": x, "y": y})

        nombre_creado, campo_creado = crear_campo(
            f"{nombre_nuevo}: ({expr_P}, {expr_Q})",
            P_expr,
            Q_expr
        )

        # Se agrega al diccionario global de campos
        FIELDS[nombre_creado] = campo_creado
        st.sidebar.success(f"Campo '{nombre_creado}' añadido ✨")
    except Exception as e:
        st.sidebar.error(f"Error al interpretar P y Q:\n{e}")

# -------- Puntos extremos A y B --------
st.sidebar.markdown("---")
st.sidebar.write("**Puntos extremos** (A y B)")
Ax = st.sidebar.number_input("A_x", value=0.0, step=0.1)
Ay = st.sidebar.number_input("A_y", value=0.0, step=0.1)
Bx = st.sidebar.number_input("B_x", value=1.0, step=0.1)
By = st.sidebar.number_input("B_y", value=1.0, step=0.1)
A = np.array([Ax, Ay])
B = np.array([Bx, By])

# -------- Parámetros de la trayectoria 2 --------
st.sidebar.markdown("---")
tray_sel = st.sidebar.selectbox(
    "Trayectoria 2",
    ["Curva parabólica (t, t^2)",
     "Familia cuadrática y=(1-a)t + a t^2"]
)
a = st.sidebar.slider("Parámetro a (solo familia)", -2.0, 2.0, 1.0, 0.1)

res = st.sidebar.slider("Resolución integración (n)", 500, 6000, 2000, 100)
dens = st.sidebar.slider("Densidad del campo (flechas)", 10, 30, 20, 1)

st.sidebar.markdown("---")
explorar_optimo = False
if (not campo["conservativo"]) and tray_sel.startswith("Familia"):
    explorar_optimo = st.sidebar.checkbox("Explorar W(a) en la familia", value=False)
else:
    st.sidebar.caption(
        "El barrido W(a) solo aplica para campos no conservativos y la familia cuadrática."
    )

# -------------------------------------------------
# CUERPO PRINCIPAL: info del campo, trabajos y gráficas
# -------------------------------------------------

# Detalles del campo
with st.expander("Detalles del campo"):
    st.write(f"**Campo seleccionado:** {campo_nombre}")
    st.write(f"**Conservativo:** {campo['conservativo']}  |  **curl** = `{campo['curl']}`")
    if campo["conservativo"]:
        if campo["potencial"] is not None:
            st.write("Tiene potencial escalar (se muestra más abajo).")
        else:
            st.write("Es conservativo (curl = 0), pero no se pudo obtener el potencial automáticamente.")

# Trayectorias
r1, dr1 = trayectoria_recta(A, B), velocidad_recta(A, B)
if tray_sel.startswith("Curva"):
    r2, dr2 = trayectoria_parabolica(A, B), velocidad_parabolica(A, B)
else:
    r2, dr2 = trayectoria_parametrica(A, B, a), velocidad_parametrica(A, B, a)

# Cálculo de trabajos
W1 = calcular_trabajo(F_np, r1, dr1, n=res)
W2 = calcular_trabajo(F_np, r2, dr2, n=res)

col1, col2 = st.columns(2)
with col1:
    st.metric("Trabajo en recta", f"{W1:.6f}")
with col2:
    st.metric("Trabajo trayectoria 2", f"{W2:.6f}")

# Campo vectorial + caminos
t_plot = np.linspace(0, 1, 500)
P1 = r1(t_plot)
P2 = r2(t_plot)

# Opcional: ajustar límites del gráfico al tamaño de las trayectorias
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

fig = quiver_2d(F_np, xlim=xlim, ylim=ylim, density=dens)
fig = add_path(fig, P1, "Recta A→B")
fig = add_path(fig, P2, "Trayectoria 2")

st.plotly_chart(fig, use_container_width=True, key="campo_2d")

# Potencial 3D (si existe)
if campo["conservativo"] and campo["potencial"] is not None:
    st.subheader("Superficie del potencial (solo campos conservativos)")
    f3d = potencial_3d(campo["potencial"], title="f(x,y)")
    st.plotly_chart(f3d, use_container_width=True, key="potencial_3d")

# Barrido W(a)
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

st.info(
    "Tip: en campos conservativos el trabajo solo depende de A y B; "
    "en campos no conservativos, depende de la forma de la trayectoria."
)
