# K-Moda · Marketing Mix Modeling

> Pipeline completo de Marketing Mix Modeling (MMM) para K-Moda: cuantificación del impacto publicitario por canal con intervalos de credibilidad bayesianos, atribución de ventas y optimización estratégica del presupuesto con dashboard interactivo.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ElasticNet-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Contexto

K-Moda es una cadena de moda con presencia en 10 ciudades españolas e inversión publicitaria de ~11 M€ anuales distribuidos en 8 canales (Paid Search, Social Paid, Video Online, Display, Email CRM, Radio Local, Exterior y Prensa).

El objetivo del proyecto es responder a una pregunta de negocio concreta:

> *¿Cuánto contribuye cada euro invertido en publicidad a las ventas? ¿Cómo debería redistribuirse el presupuesto para maximizar el retorno?*

En un entorno post-cookie donde el tracking individual ya no es viable, el MMM es la herramienta que permite medir el impacto de los medios sin datos de usuario.

---

## Demo

🚀 **[Abrir dashboard en Streamlit](https://kmoda-mmm.streamlit.app)**

---

## Resultados del modelo

### Comparativa predictiva

| Modelo | MAPE Train | MAPE Test | R² Train | R² Test |
|---|---|---|---|---|
| Ridge | 5.42% | 8.16% | 0.931 | -0.41 |
| Lasso | 5.68% | 8.10% | 0.927 | -0.43 |
| ElasticNet | 5.53% | **7.95%** | 0.930 | -0.39 |
| **Bayesiano** | — | **8.5%** | — | -0.52 |

> El R² negativo en test (todos los modelos) refleja un cambio de régimen estadístico en 2024: la volatilidad de ventas es ~10× mayor que en los años de entrenamiento. El MAPE es la métrica interpretable de referencia.

**Modelo seleccionado: Bayesiano analítico** — ROAS positivos garantizados en todos los canales + intervalos de credibilidad al 90%.

### ROAS por canal (Modelo Bayesiano)

| Canal | ROAS media | HDI 90% | Señal |
|---|---|---|---|
| Exterior | **18.73×** | [8.0 – 29.1] | Alta |
| Radio Local | 4.25× | [0.0 – 15.4] | Media |
| Social Paid | 3.75× | [0.0 – 8.5] | Media |
| Paid Search | 2.86× | [0.0 – 6.4] | Media |
| Display | 1.75× | [0.0 – 8.2] | Baja |
| Prensa | 1.70× | [0.0 – 8.8] | Baja |
| Email CRM | 1.40× | [0.0 – 8.5] | Baja |
| Video Online | 0.39× | [0.0 – 2.9] | Sin señal |

### Atribución de ventas

| Componente | Contribución media | Share |
|---|---|---|
| Base orgánica | 2.38 M EUR/semana | **74.1%** |
| Exterior | 0.50 M EUR/semana | 15.4% |
| Social Paid | 0.14 M EUR/semana | 4.4% |
| Paid Search | 0.14 M EUR/semana | 4.3% |
| Radio Local | 0.05 M EUR/semana | 1.5% |
| Display / Prensa / Email / Video | — | ~0% (β posterior = 0) |

### Optimización de presupuesto

Con el mismo presupuesto (~212.000 EUR/semana), la redistribución óptima según mROI logra un **+81.5% en ventas atribuidas**:

| Canal | Inversión actual | Inversión óptima | Cambio |
|---|---|---|---|
| Exterior | 25.679 € | 64.198 € | **+150%** |
| Social Paid | 37.178 € | 48.231 € | +30% |
| Paid Search | 47.302 € | 14.191 € | -70% |
| Radio Local | 23.515 € | 7.054 € | -70% |

---

## Pipeline

```
01_etl.ipynb                  → Limpieza, rollup semanal y auditoría de datos
02_eda.ipynb                  → Exploración y análisis descriptivo
03_feature_engineering.ipynb  → Lag + Adstock por canal, variables exógenas
04_modelos_clasicos.ipynb     → Ridge, Lasso, ElasticNet con validación temporal
05_bayesiano_mmm.ipynb        → Regresión bayesiana analítica (posterior conjugado)
06_seleccion_modelo.ipynb     → Comparativa sistemática y selección del modelo final
07_atribucion.ipynb           → Descomposición de ventas: base + contribución por canal
08_budget_simulator.ipynb     → Optimizador de presupuesto con linprog (5 escenarios)
09_dashboard_streamlit.py     → Dashboard interactivo completo
```

---

## Tecnología

| Categoría | Librerías |
|---|---|
| Datos | pandas, numpy, pyarrow |
| Modelado | scikit-learn (Ridge/Lasso/ElasticNet), scipy (linprog) |
| Bayesiano | numpy + scipy — posterior analítico conjugado (sin MCMC) |
| Visualización | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Entorno | Python 3.13, venv |

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/PabloJimenezGirardeau/kmoda-mmm.git
cd kmoda-mmm

# Crear entorno virtual
python -m venv venv_kmoda
venv_kmoda\Scripts\activate   # Windows
# source venv_kmoda/bin/activate  # macOS/Linux

# Instalar dependencias
pip install -r requirements.txt
```

> **Nota:** los datos originales (CSV) no están incluidos en el repositorio. Colócalos en la raíz del proyecto antes de ejecutar los notebooks.

---

## Uso

### Ejecutar el pipeline completo

Abrir y ejecutar los notebooks en orden (01 → 08) desde VS Code o Jupyter.

### Lanzar el dashboard

```bash
streamlit run 09_dashboard_streamlit.py
```

O usar el launcher incluido: `lanzar_dashboard.bat` (Windows).

---

## Estructura del proyecto

```
kmoda-mmm/
├── 01_etl.ipynb
├── 02_eda.ipynb
├── 03_feature_engineering.ipynb
├── 04_modelos_clasicos.ipynb
├── 05_bayesiano_mmm.ipynb
├── 06_seleccion_modelo.ipynb
├── 07_atribucion.ipynb
├── 08_budget_simulator.ipynb
├── 09_dashboard_streamlit.py
├── lanzar_dashboard.bat
├── data/                    # Parquets intermedios (generados por el pipeline)
├── img/                     # Todas las visualizaciones generadas
└── requirements.txt
```

---

## Metodología

### Transformación Adstock

Cada canal de inversión se transforma con un decay exponencial para capturar el efecto de carry-over publicitario:

```
Adstock_t = Inv_t + α · Adstock_{t-1}
```

donde `α ∈ [0, 1]` es el parámetro de retención calibrado por canal.

### Modelo Bayesiano Analítico

Regresión lineal con prior Normal conjugado → posterior en forma cerrada (sin MCMC):

```
Posterior: Σₙ = (Σ₀⁻¹ + X'X/σ²)⁻¹
           μₙ = Σₙ · (X'y/σ²)
```

Las muestras con `β_canal < 0` se descartan (prior de positividad).

### Optimizador de Presupuesto

Programación lineal (scipy HiGHS) con restricciones `floor=30%` / `ceiling=250%` por canal:

```
max  Σ ROAS_i · x_i
s.t. Σ x_i = Budget
     0.30 · inv_i ≤ x_i ≤ 2.50 · inv_i
```

---

## Autor

**Pablo Jiménez** · 3º Ingeniería Matemática · UAX  
[pablojimgir@gmail.com](mailto:pablojimgir@gmail.com)
