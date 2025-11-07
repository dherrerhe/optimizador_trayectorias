import streamlit as st
import numpy as np
from campos import FIELDS
from integrales import (
    integral_de_linea, r_recta, dr_recta_dt,
    r_parabola, dr_parabola_dt, r_familia, dr_familia_dt
)
from visualizacion import quiver_2d, add_path, potencial_3d, plot_W_vs_a

st.set_page_config(page_title="Optimizador de Trayectorias", layout="wide")
st.title("🧭 Optimizador de Trayectorias — Conservativos vs No Conservativos")

# --------- Sidebar: parámetros ---------
campo_nombre = st.sidebar.selectbox("Campo vectorial", list(FIELDS.keys()))
campo = FIELDS[campo_nombre]
F_np = campo["np"]

st.sidebar.markdown("---")
st.sidebar.write("**Puntos extremos** (recomendado A=(0,0), B=(1,1) para las curvas predefinidas)")
Ax = st.sidebar.number_input("A_x", value=0.0, step=0.1)
Ay = st.sidebar.number_input("A_y", value=0.0, step=0.1)
Bx = st.sidebar.number_input("B_x", value=1.0, step=0.1)
By = st.sidebar.number_input("B_y", value=1.0, step=0.1)
A = np.array([Ax, Ay]); B = np.array([Bx, By])

st.sidebar.markdown("---")
tray_sel = st.sidebar.selectbox("Trayectoria 2", ["Curva parabólica (t, t^2)", "Familia cuadrática y=(1-a)t + a t^2"])
a = st.sidebar.slider("Parámetro a (solo familia)", -2.0, 2.0, 1.0, 0.1)

res = st.sidebar.slider("Resolución integración (n)", 500, 6000, 2000, 100)
dens = st.sidebar.slider("Densidad del campo (flechas)", 10, 30, 20, 1)

st.sidebar.markdown("---")
opt_scan = st.sidebar.checkbox("Explorar W(a) en la familia", value=False)

# --------- Información del campo ---------
with st.expander("Detalles del campo"):
    st.write(f"**Conservativo:** {campo['conservativo']}  |  **curl** = `{campo['curl']}`")
    if campo["conservativo"]:
        st.write("Tiene potencial: f(x,y) = x² + y²")

# --------- Construcción de trayectorias ---------
r1, dr1 = r_recta(A, B), dr_recta_dt(A, B)
if tray_sel.startswith("Curva"):
    r2, dr2 = r_parabola(A, B), dr_parabola_dt(A, B)
else:
    r2, dr2 = r_familia(A, B, a), dr_familia_dt(A, B, a)

# --------- Cálculo de trabajos ---------
W1 = integral_de_linea(F_np, r1, dr1, n=res)
W2 = integral_de_linea(F_np, r2, dr2, n=res)

col1, col2 = st.columns(2)
with col1:
    st.metric("Trabajo en recta", f"{W1:.6f}")
with col2:
    st.metric("Trabajo trayectoria 2", f"{W2:.6f}")

# --------- Visualizaciones 2D ---------
fig = quiver_2d(F_np, density=dens)
# Muestras para ploteo de caminos
t_plot = np.linspace(0, 1, 500)
P1 = r1(t_plot); P2 = r2(t_plot)
fig = add_path(fig, P1, "Recta A→B")
fig = add_path(fig, P2, "Trayectoria 2")

st.plotly_chart(fig, use_container_width=True)

# --------- Potencial 3D (solo si conservativo) ---------
if campo["conservativo"] and campo["potencial"] is not None:
    st.subheader("Superficie del potencial (solo campos conservativos)")
    f3d = potencial_3d(campo["potencial"], title="f(x,y) = x² + y²")
    st.plotly_chart(f3d, use_container_width=True)

# --------- Exploración W(a) ---------
if opt_scan and tray_sel.startswith("Familia"):
    st.subheader("Barrido del parámetro a")
    a_vals = np.linspace(-2.0, 2.0, 81)
    W_vals = []
    for av in a_vals:
        rA, dA = r_familia(A, B, av), dr_familia_dt(A, B, av)
        W_vals.append(integral_de_linea(F_np, rA, dA, n=res))
    st.plotly_chart(plot_W_vs_a(a_vals, W_vals), use_container_width=True)

st.info("Tip: en campos conservativos el trabajo depende solo de A y B; en no conservativos, depende de la forma del camino.")
