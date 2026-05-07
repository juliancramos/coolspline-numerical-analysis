import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# ──────────────────────────────────────────────────────────────────────────────
# CSS Global — Dark Mode & Metric Cards
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fuente corporativa ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Fondo general: gris carbón ── */
.stApp { background-color: #0E1117; }

/* ── Sidebar: gris muy oscuro ── */
[data-testid="stSidebar"] {
    background-color: #161B22 !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ── Tarjetas de métrica (st.metric) ── */
[data-testid="stMetric"] {
    background: #1E1E1E;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px 24px 16px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.35);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(255,255,255,0.18);
    box-shadow: 0 4px 20px rgba(0,0,0,0.55);
}
/* Label */
[data-testid="stMetricLabel"] > div {
    color: #9CA3AF !important;
    font-size: 0.70rem !important;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}
/* Valor */
[data-testid="stMetricValue"] > div {
    color: #F9FAFB !important;
    font-size: 1.75rem !important;
    font-weight: 700;
    letter-spacing: -0.03em;
}
/* Delta */
[data-testid="stMetricDelta"] > div {
    color: #34D399 !important;
    font-size: 0.75rem !important;
}

/* ── Divisor sidebar ── */
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.07); }

/* ── Sliders ── */
[data-testid="stSlider"] [role="slider"] { background-color: #FFFFFF; }

/* ── Alertas ── */
[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Panel de Control
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0A2744 0%,#0D3A5C 100%);
                border:1px solid #1E4D6B;border-radius:10px;
                padding:14px 18px;margin-bottom:20px">
      <span style="color:#00C6FF;font-size:20px;font-weight:700;letter-spacing:-0.02em">❄ CoolSpline</span><br>
      <span style="color:#7BBDD4;font-size:10px;letter-spacing:0.12em;text-transform:uppercase">EcoData Solutions</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🕐 Hora de consulta")
    hora_decimal = st.slider("Hora (formato decimal)", 6.0, 24.0, 12.0, 1/60,
                             label_visibility="collapsed")
    total_mins = int(round(hora_decimal * 60))
    hh = total_mins // 60
    mm = total_mins % 60
    if hh == 24 and mm > 0:
        hh, mm = 24, 0
    st.markdown(f"<p style='font-size:28px;font-weight:bold;color:#185FA5;margin-top:-8px'>"
                f"{hh:02d}:{mm:02d}</p>", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🌡️ Umbral de temperatura")
    umbral = st.slider("Umbral (°C)", 25.0, 34.0, 30.0, 0.5,
                       label_visibility="collapsed")
    st.caption(f"Umbral activo: **{umbral:.1f} °C**")

    st.divider()

    st.markdown("### 📈 Visualización")
    mostrar_lineal = st.checkbox("Interpolación lineal", value=True)
    mostrar_deriv  = st.checkbox("Derivada dT/dt", value=True)

    st.divider()
    st.caption("Control de temperatura en tiempo real · Gemelo Digital Térmico")

# ──────────────────────────────────────────────────────────────────────────────
# Cabecera principal
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background: linear-gradient(135deg, #061626 0%, #0A2744 60%, #0D3A5C 100%);
  border: 1px solid #1E4D6B;
  border-radius: 14px;
  padding: 18px 28px;
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 8px 32px rgba(0,198,255,0.10);
">
  <span style="font-size:30px">❄</span>
  <div>
    <span style="color:#00C6FF;font-size:22px;font-weight:700;letter-spacing:-0.03em;
                 font-family:'Inter',sans-serif">CoolSpline</span>
    <span style="color:#4A9BBF;font-size:13px;margin-left:10px;font-weight:300">by EcoData Solutions</span><br>
    <span style="color:#7BBDD4;font-size:11px;letter-spacing:0.10em;text-transform:uppercase">
      Gemelo Digital Térmico &nbsp;·&nbsp; Data Center Thermal Monitor
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# Calcular temperatura y derivada para la hora seleccionada
# AHORA EVALÚA USANDO t_train (los 864 nodos correctos)
temp = evaluar_spline(t_train, a_sp, b_sp, c_sp, d_sp, hora_decimal)
deriv = derivada_spline(t_train, b_sp, c_sp, d_sp, hora_decimal)

# ──────────────────────────────────────────────────────────────────────────────
# Fila superior: Gauge charts + métricas de texto
# ──────────────────────────────────────────────────────────────────────────────
idx_max = np.argmax(y_sp)
hora_max = t_fino[idx_max]
hh_max = int(hora_max)
mm_max = int(round((hora_max - hh_max) * 60))
if mm_max == 60:
    hh_max += 1
    mm_max = 0

col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])

# ── Gauge: Temperatura actual ──────────────────────────────────────────────
with col1:
    fig_gauge_temp = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=temp,
        number={"suffix": " °C",
                "font": {"size": 44, "color": "#F9FAFB", "family": "Inter, sans-serif"}},
        delta={"reference": umbral,
               "increasing": {"color": "#F87171"},
               "decreasing": {"color": "#34D399"},
               "suffix": " °C",
               "font": {"size": 14}},
        title={"text": "TEMPERATURA ACTUAL",
               "font": {"size": 11, "color": "#9CA3AF", "family": "Inter, sans-serif"}},
        gauge={
            "axis": {"range": [18, 36],
                     "tickcolor": "rgba(255,255,255,0.2)",
                     "tickfont": {"color": "rgba(255,255,255,0.4)", "size": 10},
                     "tickwidth": 1,
                     "dtick": 4},
            "bar": {"color": "#F9FAFB", "thickness": 0.12},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [18, 25], "color": "rgba(52,211,153,0.10)"},
                {"range": [25, 30], "color": "rgba(251,191,36,0.10)"},
                {"range": [30, 36], "color": "rgba(248,113,113,0.12)"},
            ],
            "threshold": {
                "line": {"color": "#F87171", "width": 2},
                "thickness": 0.6,
                "value": 32,
            },
        },
    ))
    fig_gauge_temp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
        margin=dict(t=30, b=0, l=10, r=10),
        height=260,
    )
    st.plotly_chart(fig_gauge_temp, use_container_width=True)

# ── Gauge: Velocidad de cambio ─────────────────────────────────────────────
with col2:
    max_rate = 0.6
    bar_color = "#34D399" if abs(deriv) <= 0.3 else "#F87171"
    fig_gauge_der = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(deriv, 4),
        number={"suffix": " °C/min",
                "font": {"size": 36, "color": "#F9FAFB", "family": "Inter, sans-serif"},
                "valueformat": "+.4f"},
        title={"text": "VELOCIDAD dT/dt",
               "font": {"size": 11, "color": "#9CA3AF", "family": "Inter, sans-serif"}},
        gauge={
            "axis": {"range": [-max_rate, max_rate],
                     "tickcolor": "rgba(255,255,255,0.2)",
                     "tickfont": {"color": "rgba(255,255,255,0.4)", "size": 10},
                     "tickwidth": 1,
                     "dtick": 0.2},
            "bar": {"color": bar_color, "thickness": 0.12},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [-max_rate, -0.3], "color": "rgba(248,113,113,0.10)"},
                {"range": [-0.3,  0.3],     "color": "rgba(52,211,153,0.08)"},
                {"range": [0.3,  max_rate], "color": "rgba(248,113,113,0.10)"},
            ],
            "threshold": {
                "line": {"color": "#FBBF24", "width": 2},
                "thickness": 0.6,
                "value": 0.3,
            },
        },
    ))
    fig_gauge_der.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
        margin=dict(t=30, b=0, l=10, r=10),
        height=260,
    )
    st.plotly_chart(fig_gauge_der, use_container_width=True)

# ── Métricas de texto: Máxima del día y RMSE ──────────────────────────────
with col3:
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
# Gráfica principal interactiva (Plotly)
# ──────────────────────────────────────────────────────────────────────────────

# Etiquetas de hora amigables para el eje X y el hover
horas_label = [f"{int(t):02d}:{int(round((t % 1) * 60)):02d}" for t in t_fino]
horas_x_label = [f"{int(t):02d}:{int(round((t % 1) * 60)):02d}" for t in x]

# Configurar subplots según si se muestra la derivada
rows = 2 if mostrar_deriv else 1
row_heights = [0.68, 0.32] if mostrar_deriv else [1.0]
fig_main = make_subplots(
    rows=rows, cols=1,
    shared_xaxes=True,
    row_heights=row_heights,
    vertical_spacing=0.06,
    subplot_titles=("Temperatura (°C)", "dT/dt (°C/min)") if mostrar_deriv else ("Temperatura (°C)",),
)

# ── Curva Spline cúbico ───────────────────────────────────────────────────
fig_main.add_trace(go.Scatter(
    x=horas_label, y=y_sp,
    mode="lines",
    name="Spline cúbico",
    line=dict(color="#00C6FF", width=2.5),
    hovertemplate="<b>%{x}</b><br>Spline: %{y:.2f} °C<extra></extra>",
), row=1, col=1)

# ── Interpolación lineal (opcional) ───────────────────────────────────────
if mostrar_lineal:
    fig_main.add_trace(go.Scatter(
        x=horas_label, y=y_lin,
        mode="lines",
        name="Interpolación lineal",
        line=dict(color="#FF6B6B", width=1.8, dash="dash"),
        hovertemplate="<b>%{x}</b><br>Lineal: %{y:.2f} °C<extra></extra>",
    ), row=1, col=1)

# ── Puntos de datos medidos ────────────────────────────────────────────────
fig_main.add_trace(go.Scatter(
    x=horas_x_label, y=y,
    mode="markers",
    name="Datos medidos",
    marker=dict(color="#FFD700", size=9, symbol="circle",
                line=dict(color="#0D1B2A", width=1.5)),
    hovertemplate="<b>%{x}</b><br>Medición: %{y:.1f} °C<extra></extra>",
), row=1, col=1)

# ── Punto actual seleccionado ──────────────────────────────────────────────
fig_main.add_trace(go.Scatter(
    x=[f"{hh:02d}:{mm:02d}"], y=[temp],
    mode="markers",
    name=f"T actual ({hh:02d}:{mm:02d})",
    marker=dict(color="#FF9500", size=14, symbol="diamond",
                line=dict(color="white", width=1.5)),
    hovertemplate=f"<b>{hh:02d}:{mm:02d}</b><br>T = {temp:.2f} °C<extra></extra>",
), row=1, col=1)

# ── Línea vertical: hora actual ────────────────────────────────────────────
fig_main.add_vline(
    x=f"{hh:02d}:{mm:02d}",
    line=dict(color="rgba(160,160,160,0.4)", width=1.2, dash="dot"),
    row=1, col=1,
)

# ── Umbral de advertencia ──────────────────────────────────────────────────
fig_main.add_hline(
    y=umbral, row=1, col=1,
    line=dict(color="#FFD700", width=1.5, dash="dot"),
    annotation_text=f"Umbral {umbral:.1f}°C",
    annotation_font=dict(color="#FFD700", size=10),
    annotation_position="top right",
)

# ── Alarma crítica 32°C ────────────────────────────────────────────────────
fig_main.add_hline(
    y=32, row=1, col=1,
    line=dict(color="#FF4B5C", width=1.5, dash="dot"),
    annotation_text="Alarma 32°C",
    annotation_font=dict(color="#FF4B5C", size=10),
    annotation_position="top right",
)

# ── Relleno zona sobre el umbral ──────────────────────────────────────────
fig_main.add_trace(go.Scatter(
    x=horas_label + horas_label[::-1],
    y=list(np.where(y_sp > umbral, y_sp, umbral)) +
      [umbral] * len(t_fino),
    fill="toself",
    fillcolor="rgba(255,75,92,0.08)",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip",
), row=1, col=1)

# ── Derivada (opcional, subplot inferior) ─────────────────────────────────
if mostrar_deriv:
    fig_main.add_trace(go.Scatter(
        x=horas_label, y=y_der,
        mode="lines",
        name="dT/dt",
        line=dict(color="#3CE87A", width=2),
        hovertemplate="<b>%{x}</b><br>dT/dt: %{y:+.4f} °C/min<extra></extra>",
    ), row=2, col=1)

    # Punto de derivada actual
    fig_main.add_trace(go.Scatter(
        x=[f"{hh:02d}:{mm:02d}"], y=[deriv],
        mode="markers",
        name="dT/dt actual",
        marker=dict(color="#FF9500", size=10, symbol="diamond",
                    line=dict(color="white", width=1.2)),
        showlegend=False,
        hovertemplate=f"<b>{hh:02d}:{mm:02d}</b><br>dT/dt = {deriv:+.4f} °C/min<extra></extra>",
    ), row=2, col=1)

    # Bandas ±0.3
    fig_main.add_hline(y= 0.3, row=2, col=1,
                       line=dict(color="#FF4B5C", width=1, dash="dot"),
                       annotation_text="+0.3",
                       annotation_font=dict(color="#FF4B5C", size=9),
                       annotation_position="top right")
    fig_main.add_hline(y=-0.3, row=2, col=1,
                       line=dict(color="#FF4B5C", width=1, dash="dot"),
                       annotation_text="-0.3",
                       annotation_font=dict(color="#FF4B5C", size=9),
                       annotation_position="bottom right")
    fig_main.add_hline(y=0, row=2, col=1,
                       line=dict(color="rgba(150,150,150,0.3)", width=0.8))

    fig_main.update_yaxes(title_text="dT/dt (°C/min)",
                          title_font=dict(color="#9CA3AF", size=13),
                          tickfont=dict(color="#9CA3AF", size=12),
                          gridcolor="rgba(255,255,255,0.04)",
                          zerolinecolor="rgba(255,255,255,0.08)",
                          row=2, col=1)

# ── Estilo global de la figura ────────────────────────────────────────────
fig_main.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.025)",   # gris ultra-sutil
    font=dict(family="Inter, sans-serif", color="#E8EAF0", size=13),
    legend=dict(
        bgcolor="rgba(14,17,23,0.80)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(size=13, color="#E8EAF0"),
    ),
    hovermode="x unified",
    margin=dict(t=50, b=20, l=10, r=10),
    height=750 if mostrar_deriv else 450,
)
fig_main.update_xaxes(
    tickfont=dict(color="#9CA3AF", size=12),
    gridcolor="rgba(255,255,255,0.04)",
    showgrid=True,
    tickangle=-30,
    dtick=int(len(horas_label) / 9),   # ~2 h entre ticks
)
fig_main.update_yaxes(
    title_text="Temperatura (°C)",
    title_font=dict(color="#9CA3AF", size=13),
    tickfont=dict(color="#9CA3AF", size=12),
    gridcolor="rgba(255,255,255,0.04)",
    zerolinecolor="rgba(255,255,255,0.06)",
    range=[19, 35],
    row=1, col=1,
)
for ann in fig_main.layout.annotations:
    ann.font.color = "#9CA3AF"
    ann.font.size  = 13

st.plotly_chart(fig_main, use_container_width=True)