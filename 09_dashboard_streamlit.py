#!/usr/bin/env python
# 09_dashboard_streamlit.py — K-Moda MMM Dashboard · Fase 9
# Lanzar: streamlit run 09_dashboard_streamlit.py

import warnings; warnings.filterwarnings('ignore')
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog

# ═════════════════════════════════════════════════════════════════════
# 1 · CONFIG
# ═════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="K·Moda · Marketing Mix Model",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════
# 2 · PALETA
# ═════════════════════════════════════════════════════════════════════
BG         = "#0A0A0B"   # Fondo principal (casi negro)
SURFACE    = "#141416"   # Tarjetas
SURFACE_2  = "#1C1C1F"   # Tarjetas elevadas
BORDER     = "#2A2A2E"   # Bordes sutiles
GOLD       = "#D4AF37"   # Dorado principal
GOLD_SOFT  = "#B8962E"   # Dorado oscurecido
GOLD_LIGHT = "#E8C76A"   # Dorado claro
WHITE      = "#FFFFFF"
OFF_WHITE  = "#EDEDED"
MUTED      = "#8A8A90"   # Texto secundario
POSITIVE   = "#4ADE80"
NEGATIVE   = "#F87171"

# Paleta de canales — tonos dorados/cálidos + algunos contrastes
CANAL_COLOR = {
    'Exterior':     '#D4AF37',
    'Radio Local':  '#E8805C',
    'Social Paid':  '#F59E0B',
    'Paid Search':  '#A78BFA',
    'Display':      '#60A5FA',
    'Video Online': '#34D399',
    'Email CRM':    '#FB7185',
    'Email Crm':    '#FB7185',
    'Prensa':       '#94A3B8',
}

# ═════════════════════════════════════════════════════════════════════
# 3 · CSS GLOBAL
# ═════════════════════════════════════════════════════════════════════
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700;800;900&display=swap" rel="stylesheet">

<style>
/* ── Base ── */
html, body, [class*="css"], .main, .main * {{
    font-family: 'Inter', sans-serif !important;
}}
.stApp {{ background-color: {BG}; }}
.main .block-container {{ padding: 2rem 3rem 3rem 3rem; max-width: 1400px; }}

/* ── Streamlit hides ── */
#MainMenu, footer, .stDeployButton, header {{ visibility: hidden; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {SURFACE} !important;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {OFF_WHITE} !important; }}
[data-testid="stSidebar"] .stRadio > div {{ gap: 4px; }}
[data-testid="stSidebar"] .stRadio label {{
    padding: 10px 12px; border-radius: 8px; transition: all 0.2s;
    font-size: 0.92rem;
}}
[data-testid="stSidebar"] .stRadio label:hover {{ background: {SURFACE_2}; }}
[data-testid="stSidebar"] .stMultiSelect > div > div {{
    background: {SURFACE_2}; border: 1px solid {BORDER};
}}

/* ── Brand ── */
.brand-container {{
    text-align: center; padding: 24px 0 16px 0;
    border-bottom: 1px solid {BORDER}; margin-bottom: 20px;
}}
.brand-name {{
    font-family: 'Playfair Display', serif !important;
    font-size: 2.2rem; font-weight: 800; color: {GOLD};
    letter-spacing: 0.15em; line-height: 1;
}}
.brand-dot {{ color: {OFF_WHITE}; margin: 0 2px; }}
.brand-tag {{
    font-size: 0.68rem; color: {MUTED}; letter-spacing: 0.25em;
    margin-top: 8px; text-transform: uppercase;
}}

/* ── Hero Landing ── */
.hero {{
    background: radial-gradient(ellipse at top, {SURFACE_2} 0%, {BG} 70%);
    padding: 64px 32px 56px 32px; text-align: center;
    border-radius: 16px; margin-bottom: 32px;
    border: 1px solid {BORDER};
    position: relative; overflow: hidden;
}}
.hero::before {{
    content: ''; position: absolute; top: 0; left: 50%;
    transform: translateX(-50%); width: 220px; height: 2px;
    background: linear-gradient(90deg, transparent, {GOLD}, transparent);
}}
.hero-title {{
    font-family: 'Playfair Display', serif !important;
    font-size: 4.5rem; font-weight: 900; color: {WHITE};
    letter-spacing: 0.14em; margin: 0; line-height: 1;
}}
.hero-title .amp {{ color: {GOLD}; }}
.hero-sub {{
    font-size: 1rem; color: {MUTED}; margin-top: 18px;
    letter-spacing: 0.2em; text-transform: uppercase;
}}
.hero-desc {{
    color: {OFF_WHITE}; font-size: 1.05rem; max-width: 620px;
    margin: 28px auto 0 auto; line-height: 1.6; opacity: 0.85;
}}

/* ── Section headers ── */
.section-head {{
    display: flex; align-items: center; gap: 12px;
    margin: 28px 0 20px 0;
}}
.section-head::before {{
    content: ''; width: 4px; height: 22px; background: {GOLD}; border-radius: 2px;
}}
.section-head h2 {{
    font-size: 1.3rem; font-weight: 600; color: {WHITE};
    margin: 0; letter-spacing: 0.02em;
}}
.section-head .eyebrow {{
    font-size: 0.72rem; color: {GOLD}; letter-spacing: 0.2em;
    text-transform: uppercase; font-weight: 600;
    margin: 0 0 4px 0;
}}

/* ── Page title ── */
.page-title {{
    font-size: 2.2rem; font-weight: 700; color: {WHITE};
    letter-spacing: -0.02em; margin: 0 0 4px 0;
}}
.page-subtitle {{
    font-size: 0.95rem; color: {MUTED}; margin: 0 0 32px 0;
}}

/* ── KPI cards ── */
.kpi {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 22px 20px;
    transition: all 0.2s ease;
    position: relative; overflow: hidden;
}}
.kpi::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: {GOLD};
    transform: scaleX(0.3); transform-origin: left;
}}
.kpi:hover {{ border-color: {GOLD_SOFT}; }}
.kpi-label {{
    font-size: 0.72rem; color: {MUTED}; letter-spacing: 0.14em;
    text-transform: uppercase; font-weight: 500;
    margin-bottom: 10px;
}}
.kpi-value {{
    font-size: 2rem; font-weight: 700; color: {WHITE};
    line-height: 1.1; letter-spacing: -0.02em;
}}
.kpi-value.gold {{ color: {GOLD}; }}
.kpi-sub {{
    font-size: 0.78rem; color: {MUTED}; margin-top: 6px;
}}

/* ── Stat card (big numbers on landing) ── */
.stat {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 36px 24px;
    text-align: center;
    transition: all 0.3s ease;
}}
.stat:hover {{
    border-color: {GOLD}; transform: translateY(-2px);
}}
.stat-value {{
    font-family: 'Playfair Display', serif !important;
    font-size: 3.4rem; font-weight: 800; color: {GOLD};
    line-height: 1; margin-bottom: 12px;
}}
.stat-label {{
    font-size: 0.85rem; color: {OFF_WHITE};
    letter-spacing: 0.04em; line-height: 1.5;
}}

/* ── Channel card (ROAS per channel) ── */
.ch-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 18px 20px;
    transition: all 0.2s ease;
    position: relative; overflow: hidden;
    height: 100%;
}}
.ch-card.active {{ border-left: 3px solid; }}
.ch-card.muted {{ opacity: 0.55; }}
.ch-name {{
    font-size: 0.82rem; color: {MUTED};
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 8px; font-weight: 500;
}}
.ch-value {{
    font-size: 2rem; font-weight: 700; color: {WHITE};
    line-height: 1;
}}
.ch-value.gold {{ color: {GOLD}; }}
.ch-hdi {{
    font-size: 0.75rem; color: {MUTED};
    margin-top: 8px;
}}
.ch-bar {{
    height: 3px; background: {BORDER}; border-radius: 2px;
    margin-top: 12px; overflow: hidden;
}}
.ch-bar-fill {{ height: 100%; border-radius: 2px; transition: width 0.4s; }}

/* ── Pipeline timeline ── */
details.pipe-row {{
    border-radius: 8px; margin-bottom: 6px;
    background: {SURFACE}; border: 1px solid {BORDER};
    transition: border-color 0.2s; overflow: hidden;
}}
details.pipe-row[open] {{ border-color: {GOLD_SOFT}; }}
details.pipe-row summary {{
    display: flex; align-items: center; padding: 10px 14px;
    cursor: pointer; list-style: none; user-select: none;
}}
details.pipe-row summary::-webkit-details-marker {{ display: none; }}
details.pipe-row summary::marker {{ display: none; }}
.pipe-toggle {{
    margin-left: auto; font-size: 0.68rem; color: {MUTED};
    letter-spacing: 0.08em; transition: color 0.2s;
}}
details.pipe-row[open] .pipe-toggle {{ color: {GOLD}; }}
.pipe-detail {{
    padding: 10px 14px 12px 36px;
    color: {MUTED}; font-size: 0.81rem; line-height: 1.65;
    border-top: 1px solid {BORDER};
}}
.pipe-dot {{
    width: 8px; height: 8px; background: {GOLD};
    border-radius: 50%; margin-right: 14px; flex-shrink: 0;
    box-shadow: 0 0 0 3px {GOLD}22;
}}
.pipe-step {{ color: {GOLD}; font-weight: 600; min-width: 62px; font-size: 0.82rem; }}
.pipe-desc {{ color: {OFF_WHITE}; font-size: 0.88rem; flex: 1; }}

/* ── Dataframe styling ── */
[data-testid="stDataFrame"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}

/* ── Slider ── */
.stSlider > div > div > div {{ background: {GOLD} !important; }}
.stSlider [data-baseweb="slider"] > div:nth-child(2) {{ background: {GOLD} !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px; background: {SURFACE}; padding: 4px;
    border-radius: 10px; border: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent; color: {MUTED};
    border-radius: 6px; padding: 8px 20px;
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    background: {GOLD} !important; color: {BG} !important;
}}

/* ── Info/warning box ── */
.stAlert {{
    background: {SURFACE} !important;
    border: 1px solid {GOLD_SOFT} !important;
    border-radius: 10px !important;
    color: {OFF_WHITE} !important;
}}
.stAlert div {{ color: {OFF_WHITE} !important; }}

/* ── Botones/selects en main area ── */
.stSelectbox > div > div, .stMultiSelect > div > div {{
    background: {SURFACE} !important; border-color: {BORDER} !important;
    color: {OFF_WHITE} !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 5px; }}
::-webkit-scrollbar-thumb:hover {{ background: {GOLD_SOFT}; }}

/* ── Divider ── */
.divider {{
    border: 0; height: 1px;
    background: linear-gradient(90deg, transparent, {BORDER}, transparent);
    margin: 32px 0;
}}

/* ── Markdown text color in main area ── */
.main p, .main li {{ color: {OFF_WHITE}; }}
.main h1, .main h2, .main h3 {{ color: {WHITE}; }}
.main strong {{ color: {GOLD}; font-weight: 600; }}

/* ── Progress bar ── */
.stProgress > div > div > div > div {{ background-color: {GOLD}; }}

/* ── Plotly tooltip override ── */
.hoverlayer .hovertext {{
    background-color: {SURFACE_2} !important;
    border-color: {GOLD_SOFT} !important;
}}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════
# 4 · PLOTLY THEME BASE
# ═════════════════════════════════════════════════════════════════════
PLOTLY_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=OFF_WHITE, family='Inter', size=12),
    margin=dict(t=50, b=40, l=60, r=30),
    hoverlabel=dict(bgcolor=SURFACE_2, font_color=OFF_WHITE,
                    font_size=12, bordercolor=GOLD_SOFT),
    colorway=[GOLD, '#E8805C', '#F59E0B', '#A78BFA',
              '#60A5FA', '#34D399', '#FB7185', '#94A3B8'],
)

def style_fig(fig, title=None, height=400, show_legend=True):
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text=f'<b>{title}</b>' if title else None,
                   font=dict(size=14, color=WHITE), x=0, xanchor='left'),
        height=height,
        showlegend=show_legend,
        legend=dict(
            bgcolor='rgba(0,0,0,0)', borderwidth=0,
            font=dict(color=OFF_WHITE, size=11),
        ),
    )
    fig.update_xaxes(
        gridcolor=BORDER, gridwidth=0.5, zeroline=False,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=11),
        linecolor=BORDER,
    )
    fig.update_yaxes(
        gridcolor=BORDER, gridwidth=0.5, zeroline=False,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=11),
        linecolor=BORDER,
    )
    return fig

def hex_rgba(hx, a=1.0):
    hx = hx.lstrip('#')
    return f'rgba({int(hx[0:2],16)},{int(hx[2:4],16)},{int(hx[4:6],16)},{a})'

# ═════════════════════════════════════════════════════════════════════
# 5 · DATOS
# ═════════════════════════════════════════════════════════════════════
DATA = Path('data')

@st.cache_data
def load_data():
    df_attr  = pd.read_parquet(DATA / 'df_atribucion.parquet')
    df_roas  = pd.read_parquet(DATA / 'df_modelo_final.parquet')
    df_inv   = pd.read_parquet(DATA / 'df_inversion_clean.parquet')
    df_model = pd.read_parquet(DATA / 'df_model.parquet')
    for df in (df_attr, df_inv, df_model):
        df['semana_inicio'] = pd.to_datetime(df['semana_inicio'])
    return df_attr, df_roas, df_inv, df_model

df_attr, df_roas, df_inv, df_model = load_data()

# Escala automática
Y_SCALE = 1e6 if df_attr['y_real'].mean() < 1000 else 1.0

CANALES_INV = [c for c in df_inv.columns
               if c not in ['semana_inicio', 'anio', 'anio_iso', 'semana_iso']]
CONTRIB_COLS = [c for c in df_attr.columns
                if c.startswith('contrib_') and c != 'contrib_total_canales']
CANAL_ATTR_NAMES = [c.replace('contrib_', '') for c in CONTRIB_COLS]

TRAIN_END = pd.Timestamp('2023-12-31')
inv_media_sem = df_inv[df_inv['semana_inicio'] <= TRAIN_END][CANALES_INV].mean()
INV_TOTAL_SEM = float(inv_media_sem.sum())

def build_base_df():
    roas_dict = {r['Canal'].upper(): r['ROAS median'] for _, r in df_roas.iterrows()}
    rows = [{
        'Canal': c,
        'Inv_actual_sem': float(inv_media_sem[c]),
        'ROAS': roas_dict.get(c.upper(), 0.0),
    } for c in CANALES_INV]
    df = pd.DataFrame(rows)
    df['Ventas_attr_sem'] = df['Inv_actual_sem'] * df['ROAS']
    df['Optimizable']     = df['ROAS'] > 0
    return df

df_base = build_base_df()

# ═════════════════════════════════════════════════════════════════════
# 6 · HELPERS
# ═════════════════════════════════════════════════════════════════════
def fmt_eur(v, dec=1):
    if abs(v) >= 1e9: return f"{v/1e9:.{dec}f}B €"
    if abs(v) >= 1e6: return f"{v/1e6:.{dec}f}M €"
    if abs(v) >= 1e3: return f"{v/1e3:.{dec}f}K €"
    return f"{v:.0f} €"

def kpi(value, label, sub="", gold=False):
    cls = "kpi-value gold" if gold else "kpi-value"
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="{cls}">{value}</div>'
            f'{sub_html}</div>')

def stat_card(value, label):
    return (f'<div class="stat">'
            f'<div class="stat-value">{value}</div>'
            f'<div class="stat-label">{label}</div></div>')

def section(title, eyebrow=None):
    eb = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ''
    st.markdown(f'<div class="section-head"><div>{eb}<h2>{title}</h2></div></div>',
                unsafe_allow_html=True)

def page_header(title, subtitle=""):
    st.markdown(
        f'<h1 class="page-title">{title}</h1>'
        f'<p class="page-subtitle">{subtitle}</p>',
        unsafe_allow_html=True)

def divider():
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

def canal_color(name):
    return CANAL_COLOR.get(name, MUTED)

# ═════════════════════════════════════════════════════════════════════
# 7 · OPTIMIZACIÓN
# ═════════════════════════════════════════════════════════════════════
def optimizar_presupuesto(budget_total, floor=0.30, ceiling=2.50):
    opt_mask   = df_base['Optimizable'].values
    fixed_inv  = df_base.loc[~opt_mask, 'Inv_actual_sem'].sum()
    budget_opt = budget_total - fixed_inv
    df_opt = df_base[opt_mask].reset_index(drop=True)
    n = len(df_opt)
    result = df_base.copy()
    result['Inv_optima'] = result['Inv_actual_sem'].copy()
    if n > 0:
        floors   = np.array([floor   * v for v in df_opt['Inv_actual_sem']])
        ceilings = np.array([ceiling * v for v in df_opt['Inv_actual_sem']])
        budget_opt = float(np.clip(budget_opt, floors.sum(), ceilings.sum()))
        res = linprog(
            -df_opt['ROAS'].values,
            A_eq=np.ones((1, n)), b_eq=np.array([budget_opt]),
            bounds=list(zip(floors, ceilings)),
            method='highs',
        )
        if res.success:
            result.loc[opt_mask, 'Inv_optima'] = res.x
    result['Ventas_optimas'] = result['Inv_optima'] * result['ROAS']
    result['Delta_pct'] = ((result['Inv_optima'] - result['Inv_actual_sem'])
                           / result['Inv_actual_sem'] * 100)
    return result

@st.cache_data
def curva_respuesta():
    budgets = np.linspace(INV_TOTAL_SEM * 0.5, INV_TOTAL_SEM * 2.0, 50)
    ventas = [optimizar_presupuesto(b)['Ventas_optimas'].sum() for b in budgets]
    return budgets, np.array(ventas)

# ═════════════════════════════════════════════════════════════════════
# 8 · PLOTS
# ═════════════════════════════════════════════════════════════════════

def plot_ventas_pred(df):
    rmse = 0.458 / Y_SCALE
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['semana_inicio'], y=df['y_pred'] + rmse,
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=df['semana_inicio'], y=df['y_pred'] - rmse,
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor=hex_rgba(GOLD, 0.08),
        showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=df['semana_inicio'], y=df['y_pred'], mode='lines',
        name='Predicción del modelo',
        line=dict(color=GOLD_SOFT, width=1.5, dash='dot'),
        hovertemplate='%{x|%d %b %Y}<br>Predicción: %{y:,.2f} M€<extra></extra>'))
    fig.add_trace(go.Scatter(
        x=df['semana_inicio'], y=df['y_real'], mode='lines',
        name='Ventas reales',
        line=dict(color=GOLD, width=2.5),
        hovertemplate='%{x|%d %b %Y}<br>Real: %{y:,.2f} M€<extra></extra>'))

    # Separador train/test
    fig.add_shape(type='line', x0='2024-01-01', x1='2024-01-01',
                  y0=0, y1=1, yref='paper',
                  line=dict(color=MUTED, width=1, dash='dash'), opacity=0.5)
    fig.add_annotation(x='2024-01-01', y=1, yref='paper',
                       text=' Test 2024', showarrow=False,
                       font=dict(size=10, color=MUTED),
                       xanchor='left', yanchor='top', opacity=0.8)

    style_fig(fig, height=380)
    fig.update_layout(
        yaxis=dict(tickformat=',.1f', title='Ventas (M €)'),
        xaxis=dict(title=''),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def plot_donut_origen(df):
    base  = df['base'].sum()
    canal = df['contrib_total_canales'].sum()
    pct   = base / (base + canal) * 100
    fig = go.Figure(go.Pie(
        labels=['Base (orgánico + controles)', 'Canales pagados'],
        values=[base, canal], hole=0.7,
        marker=dict(colors=[SURFACE_2, GOLD],
                    line=dict(color=BG, width=3)),
        textinfo='none',
        hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>'))
    fig.add_annotation(
        text=f'<span style="font-size:32px;color:{GOLD};font-weight:700">{pct:.0f}%</span>'
             f'<br><span style="font-size:11px;color:{MUTED};letter-spacing:0.1em">BASE</span>',
        x=0.5, y=0.5, showarrow=False)
    style_fig(fig, height=300)
    fig.update_layout(
        legend=dict(orientation='v', x=1.05, y=0.5, yanchor='middle'),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig


def plot_decomposicion(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['semana_inicio'], y=df['base'],
        mode='lines', name='Base', stackgroup='one',
        fillcolor=hex_rgba(MUTED, 0.35),
        line=dict(width=0),
        hovertemplate='%{x|%d %b %Y}<br>Base: %{y:,.2f} M€<extra></extra>'))
    for col, cname in zip(CONTRIB_COLS, CANAL_ATTR_NAMES):
        if df[col].sum() <= 0: continue
        c = canal_color(cname)
        fig.add_trace(go.Scatter(
            x=df['semana_inicio'], y=df[col].clip(lower=0),
            mode='lines', name=cname, stackgroup='one',
            fillcolor=hex_rgba(c, 0.75), line=dict(width=0),
            hovertemplate=f'%{{x|%d %b %Y}}<br>{cname}: %{{y:,.2f}} M€<extra></extra>'))
    fig.add_trace(go.Scatter(
        x=df['semana_inicio'], y=df['y_real'], mode='lines',
        name='Ventas reales', line=dict(color=WHITE, width=2),
        hovertemplate='%{x|%d %b %Y}<br>Real: %{y:,.2f} M€<extra></extra>'))
    style_fig(fig, height=440)
    fig.update_layout(
        yaxis=dict(tickformat=',.1f', title='Ventas (M €)'),
        legend=dict(orientation='h', y=-0.18, font_size=10),
    )
    return fig


def plot_waterfall(df, año=None):
    dfw   = df[df['anio'] == año] if año else df
    label = str(año) if año else 'Período completo'
    base_v = dfw['base'].sum()
    total  = dfw['y_real'].sum()
    contribs = {cn: dfw[c].sum() for c, cn in zip(CONTRIB_COLS, CANAL_ATTR_NAMES)
                if dfw[c].sum() > 0}
    contribs_s = sorted(contribs.items(), key=lambda x: -x[1])
    x = ['Base'] + [c[0] for c in contribs_s] + ['Total ventas']
    y = [base_v] + [c[1] for c in contribs_s] + [total]
    measure = ['absolute'] + ['relative'] * len(contribs_s) + ['total']
    fig = go.Figure(go.Waterfall(
        x=x, y=y, measure=measure,
        connector=dict(line=dict(color=BORDER, width=1, dash='dot')),
        increasing=dict(marker_color=GOLD),
        decreasing=dict(marker_color=NEGATIVE),
        totals=dict(marker_color=WHITE),
        text=[f'{v:.1f}M' for v in y],
        textposition='outside', textfont=dict(size=10, color=OFF_WHITE)))
    style_fig(fig, title=f'Composición de ventas — {label}', height=400)
    fig.update_layout(yaxis=dict(tickformat='.1f', title='M €'))
    return fig


def plot_contrib_series(df, canales_sel):
    fig = go.Figure()
    for cname in canales_sel:
        col = f'contrib_{cname}'
        if col not in df.columns:
            m = next((c for c in df.columns if c.lower() == col.lower()), None)
            if m is None: continue
            col = m
        c = canal_color(cname)
        fig.add_trace(go.Scatter(
            x=df['semana_inicio'], y=df[col].clip(lower=0),
            mode='lines', name=cname, line=dict(color=c, width=2),
            hovertemplate=f'%{{x|%d %b %Y}}<br>{cname}: %{{y:,.3f}} M€<extra></extra>'))
    style_fig(fig, height=380)
    fig.update_layout(
        yaxis=dict(title='Contribución (M €)', tickformat=',.2f'),
        legend=dict(orientation='h', y=-0.18),
    )
    return fig


def plot_barras_anuales(df):
    años = sorted(df['anio'].unique())
    fig = go.Figure()
    vals_base = [df[df['anio']==a]['base'].clip(lower=0).sum()
                 / df[df['anio']==a]['y_real'].sum() * 100 for a in años]
    fig.add_trace(go.Bar(
        name='Base', x=[str(a) for a in años], y=vals_base,
        marker_color=MUTED,
        hovertemplate='Base — %{x}: %{y:.1f}%<extra></extra>'))
    for col, cname in zip(CONTRIB_COLS, CANAL_ATTR_NAMES):
        vals = []
        for a in años:
            dfa = df[df['anio']==a]
            t = dfa['y_real'].sum()
            vals.append(dfa[col].clip(lower=0).sum() / t * 100 if t > 0 else 0)
        if max(vals) > 0.1:
            fig.add_trace(go.Bar(
                name=cname, x=[str(a) for a in años], y=vals,
                marker_color=canal_color(cname),
                hovertemplate=f'{cname} — %{{x}}: %{{y:.1f}}%<extra></extra>'))
    style_fig(fig, height=360)
    fig.update_layout(
        barmode='stack',
        yaxis=dict(title='% contribución', range=[0, 100]),
        legend=dict(orientation='h', y=-0.22, font_size=10),
    )
    return fig


def plot_heatmap_seasonal(df):
    df2 = df.copy()
    df2['sw'] = df2['semana_inicio'].dt.isocalendar().week.astype(int)
    activos = [(c, cn) for c, cn in zip(CONTRIB_COLS, CANAL_ATTR_NAMES)
               if df2[c].sum() > 0]
    if not activos: return go.Figure()
    pivot = pd.DataFrame({cn: df2.groupby('sw')[c].mean()
                          for c, cn in activos}).fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values.T, x=pivot.index.tolist(), y=pivot.columns.tolist(),
        colorscale=[[0, BG], [0.3, SURFACE_2], [0.7, GOLD_SOFT], [1.0, GOLD_LIGHT]],
        hovertemplate='Semana %{x}<br>%{y}: %{z:,.3f} M€<extra></extra>',
        showscale=True,
        colorbar=dict(title=dict(text='M €', font=dict(color=MUTED)),
                       thickness=10, len=0.8,
                       tickfont=dict(color=MUTED, size=10))))
    style_fig(fig, height=300, show_legend=False)
    fig.update_layout(xaxis=dict(title='Semana del año', dtick=4),
                      yaxis=dict(title=''), margin=dict(l=110))
    return fig


def plot_gantt(df_inv_f):
    fechas = df_inv_f['semana_inicio'].tolist()
    z = df_inv_f[CANALES_INV].values.T
    fig = go.Figure(go.Heatmap(
        z=z, x=fechas, y=CANALES_INV,
        colorscale=[[0, BG], [0.01, SURFACE_2], [0.5, GOLD_SOFT], [1.0, GOLD_LIGHT]],
        hovertemplate='%{y}<br>%{x|%d %b %Y}<br>Inversión: %{z:,.0f} €<extra></extra>',
        showscale=True,
        colorbar=dict(title=dict(text='EUR', font=dict(color=MUTED)),
                       thickness=10, len=0.8,
                       tickfont=dict(color=MUTED, size=10))))
    fig.add_shape(type='line', x0='2024-01-01', x1='2024-01-01',
                  y0=-0.5, y1=len(CANALES_INV)-0.5,
                  line=dict(color=NEGATIVE, width=1.5, dash='dash'), opacity=0.7)
    style_fig(fig, height=320, show_legend=False)
    fig.update_layout(xaxis=dict(title='', tickformat='%Y'),
                      yaxis=dict(title=''), margin=dict(l=110))
    return fig


def plot_scatter_eficiencia(df_r):
    ventas_map = {cn: df_attr[c].clip(lower=0).sum() * Y_SCALE
                  for c, cn in zip(CONTRIB_COLS, CANAL_ATTR_NAMES)}
    rows = []
    for _, r in df_r.iterrows():
        c   = r['Canal']
        inv = float(inv_media_sem.get(c, 0))
        v   = ventas_map.get(c, ventas_map.get(c.replace('CRM','Crm'), 0))
        rows.append({'Canal': c, 'Inv_k': inv/1e3,
                     'ROAS': r['ROAS median'], 'Ventas_M': max(v/1e6, 0.01)})
    dfp = pd.DataFrame(rows)
    sizes = (dfp['Ventas_M'] / dfp['Ventas_M'].max() * 60 + 14).tolist()
    fig = go.Figure(go.Scatter(
        x=dfp['Inv_k'], y=dfp['ROAS'],
        mode='markers+text',
        marker=dict(size=sizes,
                    color=[canal_color(c) for c in dfp['Canal']],
                    opacity=0.85,
                    line=dict(color=BG, width=1.5)),
        text=dfp['Canal'], textposition='top center',
        textfont=dict(size=10, color=OFF_WHITE),
        hovertemplate='<b>%{text}</b><br>Inversión: %{x:.0f}K €/sem'
                      '<br>ROAS: %{y:.2f}x<extra></extra>'))
    style_fig(fig, height=380, show_legend=False)
    fig.update_layout(
        xaxis=dict(title='Inversión media semanal (K €)'),
        yaxis=dict(title='ROAS mediana'),
    )
    return fig


def plot_actual_vs_optimo(df_res):
    df_s = df_res.sort_values('ROAS', ascending=False).reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Actual', x=df_s['Canal'], y=df_s['Inv_actual_sem']/1e3,
        marker_color=MUTED, opacity=0.7,
        hovertemplate='<b>%{x}</b><br>Actual: %{y:.1f}K €<extra></extra>'))
    fig.add_trace(go.Bar(
        name='Óptimo', x=df_s['Canal'], y=df_s['Inv_optima']/1e3,
        marker_color=[canal_color(c) for c in df_s['Canal']],
        hovertemplate='<b>%{x}</b><br>Óptimo: %{y:.1f}K €<extra></extra>'))
    style_fig(fig, height=350)
    fig.update_layout(
        barmode='group',
        yaxis=dict(title='Inversión (K € / semana)'),
        legend=dict(orientation='h', y=1.05),
    )
    return fig


def plot_curva(sel_budget=None):
    budgets, ventas = curva_respuesta()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=budgets/1e3, y=ventas/1e3, mode='lines',
        name='Ventas óptimas',
        line=dict(color=GOLD, width=2.5),
        fill='tozeroy', fillcolor=hex_rgba(GOLD, 0.08),
        hovertemplate='Presupuesto: %{x:.0f}K €<br>Ventas: %{y:.0f}K €<extra></extra>'))
    idx = np.argmin(np.abs(budgets - INV_TOTAL_SEM))
    fig.add_trace(go.Scatter(
        x=[INV_TOTAL_SEM/1e3], y=[ventas[idx]/1e3],
        mode='markers', name='Presupuesto actual',
        marker=dict(color=WHITE, size=12, line=dict(color=BG, width=2))))
    if sel_budget is not None:
        idxs = np.argmin(np.abs(budgets - sel_budget))
        fig.add_trace(go.Scatter(
            x=[sel_budget/1e3], y=[ventas[idxs]/1e3],
            mode='markers', name='Seleccionado',
            marker=dict(color=POSITIVE, size=14, symbol='star',
                        line=dict(color=BG, width=1))))
    style_fig(fig, height=350)
    fig.update_layout(
        xaxis=dict(title='Presupuesto semanal (K €)'),
        yaxis=dict(title='Ventas atribuidas (K €)'),
        legend=dict(orientation='h', y=1.05),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════
# 9 · SIDEBAR
# ═════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class="brand-container">
        <div class="brand-name">K<span class="brand-dot">·</span>MODA</div>
        <div class="brand-tag">Marketing Mix Model</div>
    </div>
    """, unsafe_allow_html=True)

    pagina = st.radio(
        "Navegación",
        ["Inicio",
         "Resumen ejecutivo",
         "Rendimiento por canal",
         "Atribución",
         "Simulador de presupuesto"],
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <div style="margin-top:32px; padding-top:20px; border-top:1px solid {BORDER};
                font-size:0.72rem; color:{MUTED}; line-height:1.7; letter-spacing:0.04em;">
        <div style="color:{GOLD}; text-transform:uppercase; font-size:0.65rem;
                    letter-spacing:0.2em; margin-bottom:8px;">Modelo</div>
        Bayesiano analítico<br>
        Train 2020–2023 · Test 2024<br>
        MAPE train ≈ 8.5%
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════
# 10 · PÁGINAS
# ═════════════════════════════════════════════════════════════════════

def page_inicio():
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">K<span class="amp">·</span>M<span class="amp">·</span>M<span class="amp">·</span>M</div>
        <div class="hero-sub">Marketing Mix Model · 2020 — 2024</div>
        <div class="hero-desc">
            Modelo econométrico de atribución y optimización presupuestaria para K-Moda.
            Descompone las ventas en factores base y contribución por canal,
            cuantifica la eficiencia de cada inversión publicitaria
            y permite simular escenarios de redistribución.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Big stats
    c1, c2, c3 = st.columns(3)
    stats = [
        ('18.7x', 'ROAS del canal más eficiente<br>Exterior'),
        ('74%',   'Ventas explicadas por<br>factores base (orgánico + controles)'),
        ('8.5%',  'Error medio del modelo<br>(MAPE sobre train)'),
    ]
    for col, (v, l) in zip([c1, c2, c3], stats):
        with col: st.markdown(stat_card(v, l), unsafe_allow_html=True)

    divider()

    # Descripción + Pipeline
    col_desc, col_pipe = st.columns([1.15, 0.85])

    with col_desc:
        section("Acerca del proyecto", eyebrow="OVERVIEW")
        st.markdown("""
El **Marketing Mix Model (MMM)** mide el impacto incremental de cada canal publicitario
sobre las ventas netas, controlando por efectos estacionales y factores externos.

**Metodología aplicada:**
- Regresión lineal **Bayesiana analítica** con posterior conjugado
- Transformación **adstock** (decay geométrico) sobre la inversión
- **Intervalos de credibilidad HDI 90%** para cuantificar incertidumbre
- **Programación lineal** (HiGHS) para optimización de presupuesto

**Datos:** 262 semanas · 8 canales · 6 variables de control
""")

    with col_pipe:
        section("Pipeline", eyebrow="PIPELINE")
        fases = [
            ("Fase 1", "Análisis exploratorio",
             "EDA de 262 semanas · 8 canales y 6 variables de control · "
             "detección de outliers, missings y patrones estacionales."),
            ("Fase 2", "Limpieza y transformación",
             "Imputación de valores ausentes, normalización de escalas y "
             "construcción del dataset semanal nacional unificado."),
            ("Fase 3", "Adstock decay",
             "Transformación del gasto con decay geométrico · calibración "
             "del parámetro λ por canal para capturar el efecto retardado."),
            ("Fase 4", "Modelos clásicos",
             "Ridge y ElasticNet con validación cruzada temporal · "
             "benchmark de referencia previo al modelo bayesiano."),
            ("Fase 5", "Modelo Bayesiano",
             "Regresión bayesiana analítica con posterior conjugado · "
             "intervalos de credibilidad HDI 90% por coeficiente."),
            ("Fase 6", "Selección de modelo",
             "Comparativa MAPE / R² entre modelos · selección del modelo "
             "bayesiano como estimador final para atribución."),
            ("Fase 7", "Atribución de ventas",
             "Descomposición de ventas en base + contribución por canal · "
             "serie semanal 2020–2024 con waterfall y heatmap estacional."),
            ("Fase 8", "Simulador presupuestario",
             "Optimización lineal HiGHS maximizando ROAS · curva de "
             "respuesta y comparativa de escenarios presupuestarios."),
            ("Fase 9", "Dashboard interactivo",
             "Visualización completa con Streamlit + Plotly · "
             "5 secciones navegables y simulador en tiempo real."),
        ]
        html = "".join(
            f'<details class="pipe-row">'
            f'<summary>'
            f'<div class="pipe-dot"></div>'
            f'<div class="pipe-step">{f}</div>'
            f'<div class="pipe-desc">{d}</div>'
            f'<div class="pipe-toggle">▾</div>'
            f'</summary>'
            f'<div class="pipe-detail">{det}</div>'
            f'</details>'
            for f, d, det in fases
        )
        st.markdown(html, unsafe_allow_html=True)


def page_resumen():
    page_header("Resumen ejecutivo",
                "Indicadores clave del modelo y rendimiento agregado")

    total_ventas = df_attr['y_real'].sum() * Y_SCALE
    total_inv    = df_inv[CANALES_INV].sum().sum()
    ventas_attr  = df_attr['contrib_total_canales'].sum() * Y_SCALE
    roas_g       = ventas_attr / total_inv if total_inv > 0 else 0
    dfm          = df_attr[df_attr['anio'] <= 2023]
    mape         = np.mean(np.abs((dfm['y_real'] - dfm['y_pred']) / dfm['y_real'])) * 100
    dft          = df_attr[df_attr['anio'] == 2024]
    ss_res       = np.sum((dft['y_real'] - dft['y_pred'])**2)
    ss_tot       = np.sum((dft['y_real'] - dft['y_real'].mean())**2)
    r2_test      = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    cols = st.columns(5)
    kpis = [
        (f"{mape:.1f}%",         "MAPE",              "Train 2020–2023"),
        (f"{r2_test:+.2f}",      "R² test",           "Out-of-sample 2024"),
        (fmt_eur(total_ventas),  "Ventas totales",    "Período 2020–2024"),
        (fmt_eur(total_inv),     "Inversión total",   "Período 2020–2024"),
        (f"{roas_g:.2f}x",       "ROAS global",       "Ventas attr. / Inversión"),
    ]
    for col, (v, l, s) in zip(cols, kpis):
        with col:
            st.markdown(kpi(v, l, s, gold=(l == "ROAS global")),
                        unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    section("Ventas reales vs predicción del modelo", eyebrow="AJUSTE")
    st.plotly_chart(plot_ventas_pred(df_attr), use_container_width=True,
                    config={'displayModeBar': False})

    divider()
    c1, c2 = st.columns([0.95, 1.05])
    with c1:
        section("Origen de las ventas", eyebrow="MIX")
        st.plotly_chart(plot_donut_origen(df_attr), use_container_width=True,
                        config={'displayModeBar': False})
    with c2:
        section("Ranking de canales", eyebrow="ROAS")
        rank = df_roas.copy().sort_values('ROAS median', ascending=False).reset_index(drop=True)
        rank['Inv. total'] = rank['Canal'].map(
            lambda x: fmt_eur(df_inv[x].sum()) if x in df_inv.columns else fmt_eur(0))
        tbl = rank[['Canal', 'ROAS median', 'HDI 5%', 'HDI 95%', 'Inv. total']].copy()
        tbl.columns = ['Canal', 'ROAS mediana', 'HDI 5%', 'HDI 95%', 'Inversión total']
        st.dataframe(
            tbl.style.format({
                'ROAS mediana': '{:.2f}x',
                'HDI 5%': '{:.2f}', 'HDI 95%': '{:.2f}',
            }),
            hide_index=True, use_container_width=True, height=340,
        )

    divider()
    section("Evolución anual", eyebrow="AÑO A AÑO")

    años_all = sorted([a for a in df_attr['anio'].unique() if a >= 2020])
    yoy_rows = []
    for a in años_all:
        dfa  = df_attr[df_attr['anio'] == a]
        dfi  = df_inv[df_inv['anio'] == a] if 'anio' in df_inv.columns else df_inv
        v    = dfa['y_real'].sum() * Y_SCALE
        inv  = dfi[CANALES_INV].sum().sum()
        attr = dfa['contrib_total_canales'].sum() * Y_SCALE
        roas = attr / inv if inv > 0 else 0
        yoy_rows.append({'anio': a, 'Ventas': v, 'Inv': inv, 'ROAS': roas})
    df_yoy = pd.DataFrame(yoy_rows)
    df_yoy['delta_v'] = df_yoy['Ventas'].pct_change() * 100
    df_yoy['delta_r'] = df_yoy['ROAS'].pct_change() * 100

    year_cols = st.columns(len(años_all))
    for col, (_, row) in zip(year_cols, df_yoy.iterrows()):
        dv = row['delta_v']
        arrow_v  = ('▲' if dv > 0 else '▼') if pd.notna(dv) else ''
        color_v  = POSITIVE if (pd.notna(dv) and dv > 0) else (NEGATIVE if pd.notna(dv) else MUTED)
        delta_html = (f'<span style="color:{color_v};font-size:0.8rem;">'
                      f'{arrow_v} {abs(dv):.1f}%</span>'
                      if pd.notna(dv)
                      else f'<span style="color:{MUTED};font-size:0.8rem;">base</span>')
        with col:
            st.markdown(
                f'<div class="kpi">'
                f'<div class="kpi-label">{int(row["anio"])}</div>'
                f'<div class="kpi-value">{fmt_eur(row["Ventas"])}</div>'
                f'<div class="kpi-sub">{delta_html} · ROAS {row["ROAS"]:.2f}x</div>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    fig_yoy = go.Figure()
    años_str = [str(int(a)) for a in df_yoy['anio']]
    fig_yoy.add_trace(go.Bar(
        name='Ventas totales', x=años_str, y=df_yoy['Ventas'] / 1e6,
        marker_color=GOLD, opacity=0.85,
        hovertemplate='<b>%{x}</b><br>Ventas: %{y:.1f}M €<extra></extra>'))
    fig_yoy.add_trace(go.Scatter(
        name='ROAS global', x=años_str, y=df_yoy['ROAS'],
        mode='lines+markers', yaxis='y2',
        line=dict(color=WHITE, width=2.5),
        marker=dict(size=9, color=WHITE, line=dict(color=BG, width=2)),
        hovertemplate='<b>%{x}</b><br>ROAS: %{y:.2f}x<extra></extra>'))
    style_fig(fig_yoy, height=320)
    fig_yoy.update_layout(
        yaxis=dict(title='Ventas (M €)', tickformat='.1f'),
        yaxis2=dict(
            title='ROAS global', overlaying='y', side='right',
            showgrid=False, zeroline=False,
            tickfont=dict(color=MUTED, size=11),
            title_font=dict(color=MUTED, size=11),
        ),
        legend=dict(orientation='h', y=1.05),
    )
    st.plotly_chart(fig_yoy, use_container_width=True, config={'displayModeBar': False})


def page_roas():
    page_header("Rendimiento por canal",
                "Análisis de ROAS y eficiencia de cada canal de inversión")

    # ── Global metrics ──
    df_act = df_roas[df_roas['ROAS median'] > 0].copy()
    df_act = df_act.sort_values('ROAS median', ascending=False)
    df_zero = df_roas[df_roas['ROAS median'] == 0]

    cols = st.columns(4)
    with cols[0]:
        st.markdown(kpi(f"{len(df_act)}/{len(df_roas)}", "Canales activos",
                        "Con atribución significativa"),
                    unsafe_allow_html=True)
    with cols[1]:
        top_c = df_act.iloc[0] if len(df_act) else None
        st.markdown(kpi(f"{top_c['ROAS median']:.1f}x" if top_c is not None else "—",
                        "Mejor canal", top_c['Canal'] if top_c is not None else "—",
                        gold=True), unsafe_allow_html=True)
    with cols[2]:
        median_active = df_act['ROAS median'].median() if len(df_act) else 0
        st.markdown(kpi(f"{median_active:.2f}x", "ROAS mediana",
                        "Entre canales activos"), unsafe_allow_html=True)
    with cols[3]:
        sum_v = df_act['ROAS median'].sum() if len(df_act) else 0
        st.markdown(kpi(f"{sum_v:.1f}x", "Suma ROAS", "Eficiencia acumulada"),
                    unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # ── Canales activos como cards ──
    section("Canales con atribución significativa", eyebrow="ACTIVOS")

    max_roas = df_act['ROAS median'].max() if len(df_act) else 1
    n_active = len(df_act)
    if n_active > 0:
        cols = st.columns(n_active)
        for col, (_, row) in zip(cols, df_act.iterrows()):
            c = canal_color(row['Canal'])
            fill_pct = row['ROAS median'] / max_roas * 100
            card = f"""
            <div class="ch-card active" style="border-left-color:{c};">
                <div class="ch-name">{row['Canal']}</div>
                <div class="ch-value gold">{row['ROAS median']:.1f}x</div>
                <div class="ch-hdi">HDI 90%: [{row['HDI 5%']:.1f} — {row['HDI 95%']:.1f}]</div>
                <div class="ch-bar">
                    <div class="ch-bar-fill" style="width:{fill_pct:.0f}%;background:{c};"></div>
                </div>
            </div>
            """
            with col: st.markdown(card, unsafe_allow_html=True)

    # ── Canales sin atribución ──
    if len(df_zero) > 0:
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        section("Canales sin atribución significativa", eyebrow="INACTIVOS")
        cols = st.columns(len(df_zero))
        for col, (_, row) in zip(cols, df_zero.iterrows()):
            card = f"""
            <div class="ch-card muted">
                <div class="ch-name">{row['Canal']}</div>
                <div class="ch-value">0.0x</div>
                <div class="ch-hdi">HDI 90%: [0 — {row['HDI 95%']:.1f}]</div>
            </div>
            """
            with col: st.markdown(card, unsafe_allow_html=True)

        st.info(f"ℹ️  Los canales **{', '.join(df_zero['Canal'].tolist())}** no muestran "
                "atribución estadísticamente distinguible de cero. Posible causa: alta "
                "correlación con otros canales que se activan simultáneamente (multicolinealidad).")

    divider()

    # ── Scatter de eficiencia ──
    section("Matriz de eficiencia", eyebrow="INVERSIÓN · ROAS")
    st.markdown(f'<div style="color:{MUTED}; font-size:0.88rem; margin-bottom:12px;">'
                f'El tamaño de cada burbuja representa las ventas atribuidas totales. '
                f'Canales arriba a la izquierda: alta eficiencia con poca inversión.'
                f'</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_scatter_eficiencia(df_roas), use_container_width=True,
                    config={'displayModeBar': False})

    divider()

    # ── Tabla detallada ──
    section("Detalle completo", eyebrow="DATOS")
    tbl = df_roas[['Canal', 'ROAS median', 'ROAS media', 'HDI 5%', 'HDI 95%']].copy()
    tbl.columns = ['Canal', 'Mediana', 'Media', 'HDI 5%', 'HDI 95%']
    st.dataframe(
        tbl.style.format({'Mediana': '{:.2f}x', 'Media': '{:.2f}x',
                          'HDI 5%': '{:.2f}', 'HDI 95%': '{:.2f}'}),
        hide_index=True, use_container_width=True,
    )


def page_atribucion():
    page_header("Atribución de ventas",
                "Descomposición temporal y contribución de cada canal")

    # Filtro de años (local a esta página)
    años_disp = sorted([a for a in df_attr['anio'].unique() if a >= 2020])
    with st.expander("Filtrar por año", expanded=False):
        años_sel = st.multiselect(
            "Años", años_disp, default=años_disp, label_visibility="collapsed")
        if not años_sel:
            años_sel = años_disp
    df_f = df_attr[df_attr['anio'].isin(años_sel)].copy()

    # KPIs
    pct_base = df_f['base'].sum() / df_f['y_real'].sum() * 100 \
        if df_f['y_real'].sum() > 0 else 0
    top_col  = max(CONTRIB_COLS, key=lambda c: df_f[c].sum())
    top_name = top_col.replace('contrib_', '')
    top_pct  = df_f[top_col].sum() / df_f['y_real'].sum() * 100 \
        if df_f['y_real'].sum() > 0 else 0
    sem_max  = df_f.loc[df_f['y_real'].idxmax(), 'semana_inicio'].strftime('%d %b %Y') \
        if not df_f.empty else '—'
    ventas_top = df_f['y_real'].max() * Y_SCALE if not df_f.empty else 0

    cols = st.columns(4)
    metrics = [
        (f"{pct_base:.0f}%", "Base",                "Factores no-paid"),
        (top_name,           "Canal líder",         f"{top_pct:.1f}% de ventas"),
        (sem_max,            "Semana pico",         f"{fmt_eur(ventas_top)}"),
        (f"{len(años_sel)}", "Años seleccionados",  ", ".join(str(a) for a in años_sel)),
    ]
    for col, (v, l, s) in zip(cols, metrics):
        with col: st.markdown(kpi(v, l, s), unsafe_allow_html=True)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    section("Descomposición temporal", eyebrow="CONTRIBUCIONES")
    st.plotly_chart(plot_decomposicion(df_f), use_container_width=True,
                    config={'displayModeBar': False})

    divider()

    c1, c2 = st.columns([1, 1])
    with c1:
        años_wf = ['Período completo'] + [str(a) for a in sorted(df_f['anio'].unique())]
        año_wf  = st.selectbox("Año para waterfall", años_wf, label_visibility="collapsed")
        año_val = None if año_wf == 'Período completo' else int(año_wf)
        section("Composición waterfall", eyebrow="WATERFALL")
        st.plotly_chart(plot_waterfall(df_f, año_val), use_container_width=True,
                        config={'displayModeBar': False})
    with c2:
        section("Mix anual de atribución", eyebrow="TEMPORAL")
        st.plotly_chart(plot_barras_anuales(df_f), use_container_width=True,
                        config={'displayModeBar': False})

    divider()

    section("Series por canal", eyebrow="INDIVIDUAL")
    canales_def = [c for c in ['Exterior', 'Social Paid', 'Paid Search', 'Radio Local']
                   if c in CANAL_ATTR_NAMES]
    canales_sel = st.multiselect("Selecciona canales", CANAL_ATTR_NAMES,
                                  default=canales_def, label_visibility="collapsed")
    if canales_sel:
        st.plotly_chart(plot_contrib_series(df_f, canales_sel),
                        use_container_width=True, config={'displayModeBar': False})

    divider()

    section("Patrón estacional por canal", eyebrow="HEATMAP")
    st.plotly_chart(plot_heatmap_seasonal(df_f), use_container_width=True,
                    config={'displayModeBar': False})

    divider()

    section("Actividad de inversión publicitaria", eyebrow="GANTT")
    df_inv_f = df_inv[df_inv['anio'].isin(años_sel)].copy()
    st.plotly_chart(plot_gantt(df_inv_f), use_container_width=True,
                    config={'displayModeBar': False})


def page_simulador():
    page_header("Simulador de presupuesto",
                "Optimización en tiempo real de la asignación por canal")

    ventas_base_actual = df_base['Ventas_attr_sem'].sum()

    tab_auto, tab_manual, tab_cmp = st.tabs(
        ["Optimización automática", "Asignación manual", "Comparativa de escenarios"])

    # ── AUTOMÁTICO ────────────────────────────────────────────────────
    with tab_auto:
        st.markdown(f'<p style="color:{MUTED}; margin-bottom:16px;">'
                    f'El optimizador redistribuye hacia los canales con mayor ROAS. '
                    f'Floor 30% · Ceiling 250% por canal · Solver HiGHS.'
                    f'</p>', unsafe_allow_html=True)

        if 'sim_budget' not in st.session_state:
            st.session_state.sim_budget = int(INV_TOTAL_SEM)

        def _set_budget(mult):
            st.session_state.sim_budget = int(INV_TOTAL_SEM * mult)

        # Escenarios rápidos como pills
        section("Escenarios rápidos", eyebrow="PRESUPUESTO")
        scen = [('−20%', 0.8), ('−10%', 0.9), ('Actual', 1.0),
                ('+10%', 1.1), ('+20%', 1.2), ('+30%', 1.3)]
        pill_cols = st.columns(len(scen))
        for pc, (lbl, mult) in zip(pill_cols, scen):
            with pc:
                st.button(lbl, key=f"s_{lbl}", use_container_width=True,
                          on_click=_set_budget, args=(mult,))

        # Entrada precisa
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        cfine1, cfine2 = st.columns([0.60, 0.40])
        with cfine1:
            st.number_input(
                "Presupuesto semanal exacto (€)",
                min_value=int(INV_TOTAL_SEM * 0.5),
                max_value=int(INV_TOTAL_SEM * 2.0),
                step=1000,
                format="%d",
                key="sim_budget",
            )
        with cfine2:
            pct_vs_actual = st.session_state.sim_budget / INV_TOTAL_SEM * 100 - 100
            pct_color = GOLD if abs(pct_vs_actual) < 0.5 else (
                POSITIVE if pct_vs_actual > 0 else NEGATIVE)
            st.markdown(
                f'<div style="padding:22px 18px; background:{SURFACE}; '
                f'border:1px solid {BORDER}; border-radius:10px; text-align:center; '
                f'margin-top:28px;">'
                f'<div style="color:{MUTED}; font-size:0.68rem; letter-spacing:0.14em; '
                f'text-transform:uppercase;">vs. presupuesto actual</div>'
                f'<div style="color:{pct_color}; font-size:1.5rem; font-weight:700; margin-top:4px;">'
                f'{pct_vs_actual:+.1f}%</div></div>',
                unsafe_allow_html=True)

        budget = float(st.session_state.sim_budget)
        res = optimizar_presupuesto(budget)
        v_opt = res['Ventas_optimas'].sum()
        up    = ((v_opt - ventas_base_actual) / ventas_base_actual * 100
                 if ventas_base_actual > 0 else 0)
        roas_o = v_opt / budget if budget > 0 else 0

        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
        cols = st.columns(4)
        kpis_sim = [
            (fmt_eur(budget),   "Presupuesto",     "por semana"),
            (fmt_eur(v_opt),    "Ventas óptimas",  "por semana"),
            (f"{up:+.1f}%",     "Uplift",          "vs distribución actual"),
            (f"{roas_o:.2f}x",  "ROAS global",     "resultante"),
        ]
        for col, (v, l, s) in zip(cols, kpis_sim):
            with col: st.markdown(kpi(v, l, s, gold=(l == "Uplift")),
                                  unsafe_allow_html=True)

        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

        c_bar, c_curva = st.columns([1.1, 0.9])
        with c_bar:
            section("Reasignación por canal", eyebrow="ANTES · DESPUÉS")
            st.plotly_chart(plot_actual_vs_optimo(res), use_container_width=True,
                            config={'displayModeBar': False})
        with c_curva:
            section("Curva de respuesta", eyebrow="PRESUPUESTO → VENTAS")
            st.plotly_chart(plot_curva(budget), use_container_width=True,
                            config={'displayModeBar': False})

        section("Detalle de redistribución", eyebrow="TABLA")
        tbl = res.sort_values('ROAS', ascending=False)[
            ['Canal','Inv_actual_sem','Inv_optima','Delta_pct','ROAS','Optimizable']
        ].copy()
        tbl['Estado'] = tbl['Optimizable'].map({True:'Optimizable', False:'Fijo'})
        tbl = tbl.rename(columns={
            'Inv_actual_sem':'Actual (€/sem)', 'Inv_optima':'Óptimo (€/sem)',
            'Delta_pct':'Δ (%)', 'ROAS':'ROAS'})[
            ['Canal','Actual (€/sem)','Óptimo (€/sem)','Δ (%)','ROAS','Estado']]
        st.dataframe(
            tbl.style.format({'Actual (€/sem)':'{:,.0f}',
                              'Óptimo (€/sem)':'{:,.0f}',
                              'Δ (%)':'{:+.1f}', 'ROAS':'{:.2f}x'}),
            hide_index=True, use_container_width=True,
        )

    # ── MANUAL ────────────────────────────────────────────────────────
    with tab_manual:
        st.markdown(f'<p style="color:{MUTED}; margin-bottom:16px;">'
                    f'Edita directamente la inversión de cada canal en la tabla. '
                    f'Referencia actual: <strong>{fmt_eur(INV_TOTAL_SEM)}/semana</strong>.'
                    f'</p>', unsafe_allow_html=True)

        def _build_manual_df(inv_values):
            return pd.DataFrame({
                'Canal':     df_base['Canal'].values,
                'Actual':    df_base['Inv_actual_sem'].values,
                'ROAS':      df_base['ROAS'].values,
                'Inversión': np.asarray(inv_values, dtype=float),
            })

        if 'm_rev' not in st.session_state:
            st.session_state.m_rev = 0
        if 'm_df' not in st.session_state:
            st.session_state.m_df = _build_manual_df(df_base['Inv_actual_sem'].values)

        def _reset_actual():
            st.session_state.m_df = _build_manual_df(df_base['Inv_actual_sem'].values)
            st.session_state.m_rev += 1

        def _apply_optimum():
            r = optimizar_presupuesto(INV_TOTAL_SEM)
            st.session_state.m_df = _build_manual_df(r['Inv_optima'].values)
            st.session_state.m_rev += 1

        def _zero_all():
            st.session_state.m_df = _build_manual_df(np.zeros(len(df_base)))
            st.session_state.m_rev += 1

        b1, b2, b3 = st.columns(3)
        with b1:
            st.button("Reset a actual", on_click=_reset_actual,
                      use_container_width=True)
        with b2:
            st.button("Aplicar óptimo", on_click=_apply_optimum,
                      use_container_width=True)
        with b3:
            st.button("Vaciar todo", on_click=_zero_all,
                      use_container_width=True)

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        edited = st.data_editor(
            st.session_state.m_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                'Canal': st.column_config.TextColumn('Canal', disabled=True),
                'Actual': st.column_config.NumberColumn(
                    'Actual (€/sem)', disabled=True, format="%.0f €"),
                'ROAS': st.column_config.NumberColumn(
                    'ROAS', disabled=True, format="%.2fx"),
                'Inversión': st.column_config.NumberColumn(
                    'Nueva inversión (€/sem)',
                    min_value=0.0, step=500.0, format="%.0f €",
                    help="Edita libremente. Los cálculos se actualizan al instante."),
            },
            key=f"m_editor_{st.session_state.m_rev}",
        )

        total_m = float(edited['Inversión'].sum())
        v_m     = float((edited['Inversión'] * edited['ROAS']).sum())
        up_m    = ((v_m - ventas_base_actual) / ventas_base_actual * 100
                   if ventas_base_actual > 0 else 0)
        roas_m  = v_m / total_m if total_m > 0 else 0
        pct     = total_m / INV_TOTAL_SEM if INV_TOTAL_SEM > 0 else 0

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:{OFF_WHITE}; font-size:0.9rem; margin-bottom:6px;">'
            f'<strong style="color:{GOLD};">{fmt_eur(total_m)}</strong> '
            f'/ {fmt_eur(INV_TOTAL_SEM)} referencia '
            f'(<strong style="color:{GOLD};">{pct*100:.0f}%</strong>)</div>',
            unsafe_allow_html=True)
        st.progress(min(pct, 1.0))

        if total_m > INV_TOTAL_SEM * 1.02:
            st.warning(f"Superas el presupuesto de referencia en "
                       f"{fmt_eur(total_m - INV_TOTAL_SEM)}.")
        elif 0 < total_m < INV_TOTAL_SEM * 0.5:
            st.info("Estás muy por debajo de la referencia — revisa si es intencional.")

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        kcols = st.columns(4)
        mkpis = [
            (fmt_eur(total_m),   "Inversión total",   "por semana"),
            (fmt_eur(v_m),       "Ventas estimadas",  "por semana"),
            (f"{up_m:+.1f}%",    "Uplift",            "vs distribución actual"),
            (f"{roas_m:.2f}x",   "ROAS global",       "resultante"),
        ]
        for col, (v, l, s) in zip(kcols, mkpis):
            with col: st.markdown(kpi(v, l, s, gold=(l == "Uplift")),
                                  unsafe_allow_html=True)

        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

        dfm = df_base.copy()
        dfm['Inv_optima']     = edited['Inversión'].values
        dfm['Ventas_optimas'] = dfm['Inv_optima'] * dfm['ROAS']
        section("Distribución: actual vs. propuesta", eyebrow="COMPARATIVA")
        st.plotly_chart(plot_actual_vs_optimo(dfm), use_container_width=True,
                        config={'displayModeBar': False})

    # ── COMPARATIVA ───────────────────────────────────────────────────
    with tab_cmp:
        st.markdown(f'<p style="color:{MUTED}; margin-bottom:16px;">'
                    f'Define tres presupuestos y compara sus resultados lado a lado. '
                    f'Cada escenario se optimiza de forma independiente.'
                    f'</p>', unsafe_allow_html=True)

        SC_NAMES   = ['Escenario A', 'Escenario B', 'Escenario C']
        SC_COLORS  = [GOLD, '#60A5FA', POSITIVE]
        SC_DEFAULT = [int(INV_TOTAL_SEM * 0.9),
                      int(INV_TOTAL_SEM),
                      int(INV_TOTAL_SEM * 1.2)]

        section("Presupuestos", eyebrow="CONFIGURACIÓN")
        csc1, csc2, csc3 = st.columns(3)
        budgets_cmp = []
        for col, nm, dft, c_sc in zip([csc1, csc2, csc3], SC_NAMES, SC_DEFAULT, SC_COLORS):
            with col:
                st.markdown(
                    f'<div style="color:{c_sc}; font-size:0.72rem; font-weight:600; '
                    f'letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">'
                    f'{nm}</div>', unsafe_allow_html=True)
                b = st.number_input(
                    nm, min_value=int(INV_TOTAL_SEM * 0.5),
                    max_value=int(INV_TOTAL_SEM * 2.0),
                    value=dft, step=1000, format="%d",
                    label_visibility="collapsed", key=f"cmp_{nm}",
                )
                budgets_cmp.append(float(b))

        res_cmp = [optimizar_presupuesto(b) for b in budgets_cmp]

        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
        section("Métricas comparadas", eyebrow="RESULTADOS")

        card_cols = st.columns(3)
        for col, nm, b, r, c_sc in zip(card_cols, SC_NAMES, budgets_cmp, res_cmp, SC_COLORS):
            v    = r['Ventas_optimas'].sum()
            up   = (v - ventas_base_actual) / ventas_base_actual * 100 \
                   if ventas_base_actual > 0 else 0
            roas_sc = v / b if b > 0 else 0
            up_color = POSITIVE if up > 0 else NEGATIVE
            with col:
                st.markdown(
                    f'<div style="background:{SURFACE}; border:1px solid {c_sc}44; '
                    f'border-top:3px solid {c_sc}; border-radius:12px; padding:22px 18px;">'
                    f'<div style="color:{c_sc}; font-size:0.72rem; font-weight:600; '
                    f'letter-spacing:0.15em; text-transform:uppercase; margin-bottom:14px;">'
                    f'{nm}</div>'
                    f'<div style="color:{MUTED}; font-size:0.68rem; letter-spacing:0.1em; '
                    f'text-transform:uppercase;">Presupuesto</div>'
                    f'<div style="color:{WHITE}; font-size:1.15rem; font-weight:600; '
                    f'margin-bottom:10px;">{fmt_eur(b)}</div>'
                    f'<div style="color:{MUTED}; font-size:0.68rem; letter-spacing:0.1em; '
                    f'text-transform:uppercase;">Ventas óptimas</div>'
                    f'<div style="color:{WHITE}; font-size:1.15rem; font-weight:600; '
                    f'margin-bottom:10px;">{fmt_eur(v)}</div>'
                    f'<div style="display:flex; gap:20px;">'
                    f'<div><div style="color:{MUTED};font-size:0.68rem;text-transform:uppercase;'
                    f'letter-spacing:0.08em;">Uplift</div>'
                    f'<div style="color:{up_color};font-size:1.1rem;font-weight:700;">'
                    f'{up:+.1f}%</div></div>'
                    f'<div><div style="color:{MUTED};font-size:0.68rem;text-transform:uppercase;'
                    f'letter-spacing:0.08em;">ROAS</div>'
                    f'<div style="color:{c_sc};font-size:1.1rem;font-weight:700;">'
                    f'{roas_sc:.2f}x</div></div>'
                    f'</div></div>',
                    unsafe_allow_html=True)

        st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
        section("Asignación por canal", eyebrow="DISTRIBUCIÓN ÓPTIMA")

        canales_ord = df_base.sort_values('ROAS', ascending=False)['Canal'].tolist()
        fig_cmp = go.Figure()
        for r, nm, c_sc in zip(res_cmp, SC_NAMES, SC_COLORS):
            r_ord = r.set_index('Canal').reindex(canales_ord).reset_index()
            fig_cmp.add_trace(go.Bar(
                name=nm, x=r_ord['Canal'], y=r_ord['Inv_optima'] / 1e3,
                marker_color=c_sc, opacity=0.85,
                hovertemplate=f'<b>%{{x}}</b><br>{nm}: %{{y:.1f}}K €<extra></extra>'))
        style_fig(fig_cmp, height=360)
        fig_cmp.update_layout(
            barmode='group',
            yaxis=dict(title='Inversión óptima (K € / sem)'),
            legend=dict(orientation='h', y=1.05),
        )
        st.plotly_chart(fig_cmp, use_container_width=True, config={'displayModeBar': False})

        divider()
        section("Resumen tabular", eyebrow="TABLA")
        summary_rows = []
        for nm, b, r in zip(SC_NAMES, budgets_cmp, res_cmp):
            v  = r['Ventas_optimas'].sum()
            up = (v - ventas_base_actual) / ventas_base_actual * 100 \
                 if ventas_base_actual > 0 else 0
            summary_rows.append({
                'Escenario':      nm,
                'Presupuesto':    fmt_eur(b),
                'Ventas óptimas': fmt_eur(v),
                'Uplift':         f'{up:+.1f}%',
                'ROAS global':    f'{v/b:.2f}x' if b > 0 else '—',
                'Δ vs actual':    fmt_eur(b - INV_TOTAL_SEM),
            })
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════
# 11 · ROUTER
# ═════════════════════════════════════════════════════════════════════
{
    'Inicio':                   page_inicio,
    'Resumen ejecutivo':        page_resumen,
    'Rendimiento por canal':    page_roas,
    'Atribución':               page_atribucion,
    'Simulador de presupuesto': page_simulador,
}[pagina]()
