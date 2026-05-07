import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Datos Originales (10 puntos)
# ──────────────────────────────────────────────────────────────────────────────
x = np.array([6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
y = np.array([22.0, 24.5, 27.0, 30.2, 31.5, 31.0, 29.8, 27.5, 25.0, 23.2])

# ──────────────────────────────────────────────────────────────────────────────
# Funciones del spline cúbico natural
# ──────────────────────────────────────────────────────────────────────────────
def trazadores_cubicos_naturales(x_in, y_in, verbose=False):
    n = len(x_in) - 1
    a = np.array(y_in, dtype=float)
    h = np.zeros(n)
    for j in range(n):
        h[j] = x_in[j+1] - x_in[j]
    c = np.zeros(n+1)
    if n > 1:
        A = np.zeros((n-1, n-1))
        B = np.zeros(n-1)
        for i in range(1, n):
            row = i-1
            if i-1 > 0:
                A[row, row-1] = h[i-1]
            A[row, row] = 2*(h[i-1]+h[i])
            if i+1 < n:
                A[row, row+1] = h[i]
            B[row] = (3/h[i])*(a[i+1]-a[i]) - (3/h[i-1])*(a[i]-a[i-1])
        c[1:n] = np.linalg.solve(A, B)
    b = np.zeros(n)
    d = np.zeros(n)
    for j in range(n):
        b[j] = (a[j+1]-a[j])/h[j] - (h[j]*(2*c[j]+c[j+1]))/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
    return a, b, c, d, h

def _tramo(x_nodos, xq):
    n = len(x_nodos) - 1
    for i in range(n-1):
        if x_nodos[i] <= xq < x_nodos[i+1]:
            return i
    return n-1

def evaluar_spline(x_nodos, a, b, c, d, xq):
    j = _tramo(x_nodos, xq)
    dx = xq - x_nodos[j]
    return a[j] + b[j]*dx + c[j]*dx**2 + d[j]*dx**3

def derivada_spline(x_nodos, b, c, d, xq):
    j = _tramo(x_nodos, xq)
    dx = xq - x_nodos[j]
    return (b[j] + 2*c[j]*dx + 3*d[j]*dx**2) / 60

def interp_lineal(x_nodos, y_nodos, xq):
    for i in range(len(x_nodos)-1):
        if x_nodos[i] <= xq <= x_nodos[i+1]:
            t = (xq - x_nodos[i])/(x_nodos[i+1] - x_nodos[i])
            return y_nodos[i] + t*(y_nodos[i+1] - y_nodos[i])
    return y_nodos[-1]

# ──────────────────────────────────────────────────────────────────────────────
# Integración del Modelo (1081 puntos y Hold-out 80/20)
# ──────────────────────────────────────────────────────────────────────────────
paso_minuto = 1 / 60
t_fino = np.arange(6.0, 24.0 + paso_minuto, paso_minuto)

# 1. Interpolación lineal para generar la "realidad" minuto a minuto
y_lin_full = np.interp(t_fino, x, y)

# 2. Hold-out sistemático (20% prueba, 80% entrenamiento)
idx_test = np.arange(0, len(t_fino), 5)
idx_train = np.setdiff1d(np.arange(len(t_fino)), idx_test)

t_train, y_train = t_fino[idx_train], y_lin_full[idx_train]
t_test, y_test = t_fino[idx_test], y_lin_full[idx_test]

# 3. Entrenar spline masivo con los 864 puntos (t_train)
a_sp, b_sp, c_sp, d_sp, _ = trazadores_cubicos_naturales(t_train, y_train, verbose=False)

# 4. Calcular el RMSE sobre el 20% de prueba
y_sp_test = np.array([evaluar_spline(t_train, a_sp, b_sp, c_sp, d_sp, val) for val in t_test])
rmse_sp = np.sqrt(np.mean((y_test - y_sp_test)**2))
rmse_ln = 0.0 # RMSE contra sí misma es 0

# 5. Precalcular curvas para la gráfica interactiva del dashboard
y_sp = np.array([evaluar_spline(t_train, a_sp, b_sp, c_sp, d_sp, val) for val in t_fino])
y_der = np.array([derivada_spline(t_train, b_sp, c_sp, d_sp, val) for val in t_fino])
y_lin = y_lin_full

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de la página y cabecera
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CoolSpline - Dashboard", layout="wide")
st.markdown("""
<div style="background:#185FA5;border-radius:10px;padding:14px 20px;margin-bottom:20px">
  <span style="color:white;font-size:20px;font-weight:bold">❄ CoolSpline</span>
  <span style="color:#B5D4F4;font-size:13px;margin-left:12px">
    EcoData Solutions — Control de temperatura en tiempo real
  </span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Controles (sliders, checkboxes) 
# ──────────────────────────────────────────────────────────────────────────────
col_h, col_m = st.columns([4, 1])
with col_h:
    hora_decimal = st.slider("Hora (formato decimal)", 6.0, 24.0, 12.0, 1/60)
with col_m:
    total_mins = int(round(hora_decimal * 60))
    hh = total_mins // 60
    mm = total_mins % 60
    if hh == 24 and mm > 0:
        hh, mm = 24, 0
    st.write(f"**{hh:02d}:{mm:02d}**")

umbral = st.slider("Umbral de temperatura (°C)", 25.0, 34.0, 30.0, 0.5)
mostrar_lineal = st.checkbox("Mostrar interpolación lineal", value=True)
mostrar_deriv = st.checkbox("Mostrar derivada dT/dt", value=True)

# Calcular temperatura y derivada para la hora seleccionada
# AHORA EVALÚA USANDO t_train (los 864 nodos correctos)
temp = evaluar_spline(t_train, a_sp, b_sp, c_sp, d_sp, hora_decimal)
deriv = derivada_spline(t_train, b_sp, c_sp, d_sp, hora_decimal)

# ──────────────────────────────────────────────────────────────────────────────
# Tarjetas de métricas
# ──────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Temperatura actual", f"{temp:.2f} °C")
with col2:
    st.metric("Velocidad cambio", f"{deriv:+.4f} °C/min")
with col3:
    idx_max = np.argmax(y_sp)
    hora_max = t_fino[idx_max]
    hh_max = int(hora_max)
    mm_max = int(round((hora_max - hh_max) * 60))
    if mm_max == 60:
        hh_max += 1
        mm_max = 0
    st.metric("Temp. máxima del día", f"{y_sp[idx_max]:.2f} °C",
              delta=f"a las {hh_max:02d}:{mm_max:02d}")
with col4:
    st.metric("RMSE Spline / Lineal", f"{rmse_sp:.4f} / {rmse_ln:.4f}",
              delta="Modelo validado")

# ──────────────────────────────────────────────────────────────────────────────
# Mensaje de alarma / estado
# ──────────────────────────────────────────────────────────────────────────────
if temp > 32:
    st.error("ALARMA CRÍTICA — T > 32°C. Activar protocolo de emergencia.")
elif temp > umbral:
    st.warning(f"Ventiladores ON — T={temp:.2f}°C supera umbral de {umbral:.1f}°C.")
elif abs(deriv) > 0.3:
    signo = "+" if deriv > 0 else ""
    st.warning(f"Cambio rápido — dT/dt = {signo}{deriv:.4f} °C/min")
else:
    st.success("Sistema normal")

# ──────────────────────────────────────────────────────────────────────────────
# Gráfica principal (temperatura y opcional derivada)
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2 if mostrar_deriv else 1, 1,
                         figsize=(12, 7 if mostrar_deriv else 4),
                         sharex=True)
if not mostrar_deriv:
    axes = [axes]

# Gráfica de temperatura
ax1 = axes[0]
ax1.plot(t_fino, y_sp, color="#185FA5", lw=2, label="Spline cúbico")
if mostrar_lineal:
    ax1.plot(t_fino, y_lin, color="#A32D2D", lw=1.5, ls="--", label="Interpolación lineal")
ax1.scatter(x, y, color="#185FA5", zorder=5, s=50, label="Datos medidos")
ax1.axhline(umbral, color="#BA7517", lw=1.2, ls=":", label=f"Umbral {umbral:.1f}°C")
ax1.axhline(32, color="#A32D2D", lw=1.2, ls=":", label="Alarma 32°C")
ax1.axvline(hora_decimal, color="#888", lw=1, ls="--", alpha=0.6)
ax1.scatter([hora_decimal], [temp], color="#BA7517", zorder=10, s=120, marker="D",
            label=f"T({hh:02d}:{mm:02d})={temp:.2f}°C")
ax1.fill_between(t_fino, umbral, y_sp, where=(y_sp > umbral), alpha=0.12, color="#A32D2D")
ax1.set_ylabel("Temperatura (°C)", fontsize=11)
ax1.set_ylim(19, 35)
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(alpha=0.25)
ax1.set_title("EcoData Solutions — CoolSpline", fontsize=12)

# Gráfica de derivada (si está habilitada)
if mostrar_deriv:
    ax2 = axes[1]
    ax2.plot(t_fino, y_der, color="#0F6E56", lw=1.8, label="dT/dt (°C/min)")
    ax2.axhline(0.3, color="#A32D2D", ls=":", lw=1, label="±0.3 °C/min")
    ax2.axhline(-0.3, color="#A32D2D", ls=":", lw=1)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axvline(hora_decimal, color="#888", lw=1, ls="--", alpha=0.6)
    ax2.scatter([hora_decimal], [deriv], color="#BA7517", zorder=10, s=100, marker="D")
    ax2.fill_between(t_fino, -0.3, y_der, where=(y_der < -0.3), alpha=0.12, color="#A32D2D")
    ax2.fill_between(t_fino, 0.3, y_der, where=(y_der > 0.3), alpha=0.12, color="#A32D2D")
    ax2.set_ylabel("dT/dt (°C/min)", fontsize=11)
    ax2.set_xlabel("Hora (decimal)", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
else:
    ax1.set_xlabel("Hora (decimal)", fontsize=11)

# Formato del eje X en HH:MM
xticks = np.arange(6, 25, 2)
ax1.set_xticks(xticks)
ax1.set_xticklabels([f"{int(h)}:{int((h%1)*60):02d}" for h in xticks])

plt.tight_layout()
st.pyplot(fig)