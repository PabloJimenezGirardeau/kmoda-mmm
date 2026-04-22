# K-Moda Marketing Mix Modeling — Guía del Proyecto
**Asignatura:** Inteligencia Artificial · 3º Ingeniería Matemática, UAX  
**Autor:** Pablo Jiménez

---

## Qué es este proyecto

Pipeline completo de **Marketing Mix Modeling (MMM)** para la empresa ficticia K-Moda.  
El objetivo es demostrar con rigor estadístico que el presupuesto publicitario de ~12 M€ está bien invertido y proponer su redistribución óptima.

**Contexto narrativo:**  
Elena Torres (CMO) necesita justificar el presupuesto ante Ricardo Sanz (CFO) en un entorno post-cookie donde el tracking individual ya no es viable. El MMM es la herramienta que sustituye al attribution click-based.

---

## Entorno de trabajo

- **Python:** 3.13
- **IDE:** VS Code + extensión Jupyter
- **Gestión de entorno:** venv o conda

### Instalación

```bash
# Crear entorno
python -m venv venv_kmoda
venv_kmoda\Scripts\activate   # Windows

# Dependencias base
pip install pandas numpy scipy scikit-learn
pip install matplotlib seaborn plotly altair
pip install pyarrow fastparquet
pip install ipykernel ipywidgets jupyter
pip install streamlit

# Para el modelado bayesiano (Fase 5)
pip install pymc bambi arviz
pip install pymc-marketing   # librería MMM bayesiano de PyMC Labs — adstock + saturación + optimizador

# Registrar kernel
python -m ipykernel install --user --name=kmoda --display-name "Python (K-Moda MMM)"
```

---

## Datos disponibles

**Carpeta:** `CASOMAT_MM_07_VENTAS_LINEAS/` (no modificar nunca los originales)

| Archivo | Rol |
|---|---|
| `CASOMAT_MM_01_CLIENTES.csv` | Dimensión CRM |
| `CASOMAT_MM_02_PRODUCTOS.csv` | Dimensión producto — contiene margen objetivo medio (~67.6%) |
| `CASOMAT_MM_03_CALENDARIO.csv` | Variables exógenas — **tabla raíz para cualquier JOIN** |
| `CASOMAT_MM_04_TRAFICO_DIARIO.csv` | Variable mediadora (tráfico web y tienda) |
| `CASOMAT_MM_05_INVERSION_MEDIOS.csv` | Variable independiente X_{t,m} — semanal, por ciudad y canal |
| `CASOMAT_MM_06_PEDIDOS.csv` | Cabecera transaccional — **no usar como Y_t** |
| `CASOMAT_MM_07_VENTAS_LINEAS.csv` | **Variable dependiente Y_t** — usar siempre `venta_neta_sin_iva_eur` |

**Período:** 2020–2024  
**Ciudades:** Madrid, Barcelona, Valencia, Sevilla, Málaga, Zaragoza, Bilbao, Murcia, Palma, A Coruña  
**Canales publicitarios:** Paid Search, Social Paid, Video Online, Display, Email CRM, Radio Local, Exterior, Prensa

---

## Reglas críticas (no saltarse nunca)

1. **Variable dependiente:** siempre `venta_neta_sin_iva_eur` de VENTAS_LINEAS. Nunca el bruto de PEDIDOS.
2. **LEFT JOIN obligatorio:** la tabla raíz es siempre CALENDARIO. Los días sin ventas = 0, no eliminados.
3. **Lag antes que Adstock:** primero desplazar la serie temporal, luego aplicar el decaimiento.
4. **Coeficientes positivos:** los betas de los canales no pueden ser negativos (restricción de negocio).
5. **Split temporal sin shuffle:** TRAIN = años anteriores al periodo de test, TEST = periodo más reciente.
6. **Validación temporal:** usar TimeSeriesSplit, nunca KFold estándar.
7. **Desescalar coeficientes** antes de interpretar: β_original = β_escalado / std_feature.

---

## Metodología MMM

### Transformaciones de medios

**Lag:** desplazamiento temporal que refleja que la publicidad no tiene efecto inmediato.

$$X'_{t,m} = X_{t-L_m, m}$$

**Adstock:** efecto acumulado y decreciente de la publicidad a lo largo del tiempo.

$$A_{t,m} = X'_{t,m} + \alpha_m \cdot A_{t-1,m}$$

Los parámetros de lag y alpha deben ser justificados para cada canal según su naturaleza (digital vs. offline, branding vs. performance).

### Modelo de regresión

$$Y_t = \beta_0 + \sum_{m} \beta_m \cdot A_{t,m} + \sum_{j} \gamma_j \cdot C_{t,j} + \varepsilon_t$$

Donde $C_{t,j}$ son variables exógenas del calendario (estacionalidad, festivos, tendencia).

### Atribución

$$\text{Venta Atribuida}_m = \beta_m^{\text{orig}} \cdot \sum_t A_{t,m}$$

$$\text{Peso\%}_m = \frac{\beta_m \cdot \sum_t A_{t,m}}{\sum_k \beta_k \cdot \sum_t A_{t,k}} \times 100$$

### Curvas de saturación (rendimientos decrecientes)

La relación inversión → ventas no es lineal: doblar el presupuesto en un canal no dobla su contribución. Los rendimientos son decrecientes. La forma estándar de modelarlo es con una función de saturación aplicada sobre el adstock:

**Hill / Michaelis-Menten:**
$$S(A_{t,m}) = \frac{A_{t,m}^\alpha}{A_{t,m}^\alpha + K^\alpha}$$

**Logística:**
$$S(A_{t,m}) = \frac{L}{1 + e^{-k(A_{t,m} - x_0)}}$$

donde $K$ (o $x_0$) controla el punto de inflexión y $\alpha$ la pendiente de saturación.

Incluir saturación es especialmente relevante para el simulador de presupuesto: sin ella, la función de respuesta es lineal y siempre conviene concentrar el presupuesto en el canal con mayor beta. Con saturación aparecen rendimientos decrecientes que justifican la diversificación.

> `pymc-marketing` implementa adstock + saturación de forma nativa. En el modelado clásico (sklearn) se puede aplicar la transformación como paso adicional en el feature engineering antes de la regresión.

### Simulador (estado estacionario)

$$A_{\infty,m} = \frac{\text{Inv}_{\text{semanal},m}}{1 - \alpha_m}$$

$$\text{Contrib}_{\text{anual},m} = \beta_m \cdot A_{\infty,m} \cdot 52$$

---

## Identidad visual K-Moda

Todas las gráficas deben seguir esta paleta y filosofía. Es un criterio de calidad del proyecto.

### Paleta de colores

```python
KMODA_PALETTE = {
    # Identidad corporativa
    "gold":        "#C9A84C",   # dorado K-Moda — acento principal
    "gold_light":  "#E8D5A3",
    "charcoal":    "#2C2C2C",   # texto y ejes
    "warm_gray":   "#6B6560",
    "off_white":   "#F7F5F0",   # fondo de figura
    "white":       "#FFFFFF",

    # Canales publicitarios
    "paid_search": "#1A6B8A",
    "social_paid": "#E07B39",
    "video":       "#8B5E9E",
    "display":     "#3A9E6F",
    "email_crm":   "#D4A843",
    "radio":       "#C0504D",
    "exterior":    "#4F7CAC",
    "prensa":      "#7A7A52",

    # Semáforo
    "positive":    "#2E7D52",
    "neutral":     "#C9A84C",
    "negative":    "#B33A3A",
    "base":        "#D6CFC4",
    "incremental": "#C9A84C",
}
```

### Filosofía de visualización

- **Abundancia:** generar todas las gráficas que aporten información. Cada distribución, correlación o comparación merece su propia visualización. Más siempre es mejor que menos.
- **Coherencia:** aplicar el mismo estilo en todos los notebooks desde el inicio.
- **Librerías según contexto:**
  - Matplotlib + Seaborn → análisis exploratorio, residuos, distribuciones, heatmaps
  - Plotly → series temporales interactivas, comparativas de escenarios
  - Altair → gráficos declarativos (waterfall, scatter)
- **Títulos en español**, subtítulo técnico en inglés si aplica.
- **Guardar siempre** en `img/` con `dpi=150` y `bbox_inches="tight"`.
- El dorado `#C9A84C` se reserva para destacar el canal o métrica más relevante. Nunca como fondo.

---

## Estructura del proyecto

### Fases

El proyecto se divide en fases bien delimitadas. Cada fase tiene un único notebook o script, con una responsabilidad clara y outputs exportados a disco.

---

#### FASE 1 — `01_etl.ipynb` · ETL y Auditoría de Datos

**Objetivo:** construir la señal de ventas limpia y validar la calidad de todos los datasets.

- Cargar los 7 CSV, verificar nulos, tipos y fechas
- Rollup transaccional: agregar `venta_neta_sin_iva_eur` por semana y ciudad
- LEFT JOIN desde CALENDARIO como tabla raíz (semanas sin ventas → 0)
- Auditoría bruto vs. neto: cuantificar la sobreestimación del importe bruto
- Verificar la relación 1:N entre PEDIDOS y VENTAS_LINEAS
- Exportar `data/df_ventas_clean.parquet` y `data/df_inversion_clean.parquet`

---

#### FASE 2 — `02_eda.ipynb` · Exploración y Entendimiento de Datos

**Objetivo:** entender la estructura, estacionalidad y comportamiento de los datos antes de modelar.

- Serie temporal de ventas semanales con eventos clave marcados (COVID, iOS14, GDPR, Black Friday)
- Estacionalidad: por mes, semana, festivos
- Distribución de ventas por ciudad y canal de venta
- Inversión por canal publicitario a lo largo del tiempo
- Análisis de correlación preliminar entre inversión y ventas
- Visualizar la degradación del entorno de datos (privacidad, tracking)

---

#### FASE 3 — `03_feature_engineering.ipynb` · Construcción de Features

**Objetivo:** transformar la inversión bruta en señales de medios interpretables por el modelo.

- Agregar inversión a granularidad **semanal nacional** (sumar ciudades)
- Aplicar Lag y Adstock por canal — justificar los parámetros elegidos
- Construir variables exógenas desde CALENDARIO (tendencia, estacionalidad, festivos)
- Incorporar tráfico web y tienda como variable mediadora
- JOIN final: ventas + adstock + exógenas → `data/df_model.parquet`
- Analizar correlaciones entre adstocks y Y_t para validar las transformaciones

---

#### FASE 4 — `04_classical_models.ipynb` · Modelos de Regresión Clásicos

**Objetivo:** comparar Ridge, Lasso y ElasticNet bajo las mismas condiciones y seleccionar el mejor.

- Split temporal estricto (sin shuffle): años anteriores para TRAIN, periodo más reciente para TEST
- StandardScaler sobre features
- Entrenar y evaluar los tres modelos:
  - **Ridge** (L2): todos los coeficientes activos, shrinkage suave
  - **Lasso** (L1 puro): selección automática de canales, algunos a cero
  - **ElasticNet** (L1 + L2): combinación, control de sparsity
- Para cada uno: búsqueda de hiperparámetros con TimeSeriesSplit
- Comparar MAPE y R² en train y test
- Investigar si existe cambio de distribución entre el periodo de train y test — ¿los datos de test son estadísticamente distintos a los de train? ¿Por qué?
- Exportar el mejor modelo a `models/`

> **Reto conocido:** el periodo de test puede presentar una variabilidad muy distinta al periodo de entrenamiento. Investigar este fenómeno en profundidad antes de concluir que el modelo "falla".

---

#### FASE 5 — `05_bayesian_mmm.ipynb` · Marketing Mix Modeling Bayesiano

**Objetivo:** implementar un MMM bayesiano que aporte cuantificación de incertidumbre y curvas de saturación.

- Usar **`pymc-marketing`** como librería principal — tiene adstock (geométrico y delayed), curvas de saturación (Hill) y optimizador de presupuesto implementados y testeados
- Alternativamente, PyMC o Bambi si se quiere mayor control sobre la especificación
- Priors sobre los betas (positivos, weakly informative)
- Adstock + saturación Hill como transformaciones dentro del modelo probabilístico
- Sampling MCMC (NUTS) y diagnóstico de convergencia (R-hat, ESS, trace plots)
- Comparar los betas posteriores con los coeficientes del modelo clásico seleccionado
- Ventaja clave: intervalos de credibilidad sobre la atribución de cada canal y sobre el mROI

> **Por qué bayesiano:** en MMM la incertidumbre sobre los coeficientes es información de negocio. Decir "el mROI de Exterior es 10.4x ± 2.1x" es más honesto y más útil que un punto estimado.
>
> **Por qué `pymc-marketing`:** evita reimplementar adstock y saturación desde cero, que son las partes más delicadas del pipeline y donde es fácil introducir errores. Documentación: https://www.pymc-marketing.io

---

#### FASE 6 — `06_model_selection.ipynb` · Comparativa y Selección Final

**Objetivo:** decisión documentada sobre qué modelo usar para atribución y simulador.

- Tabla comparativa de todos los modelos entrenados (métricas, complejidad, interpretabilidad)
- Análisis de estabilidad de coeficientes entre modelos
- Justificación de la elección final
- Guardar el modelo definitivo en `models/modelo_final.pkl`

---

#### FASE 7 — `07_attribution.ipynb` · Atribución e Insights de Negocio

**Objetivo:** traducir los coeficientes del modelo en lenguaje ejecutivo.

- Calcular la venta atribuida y el peso% por canal
- Calcular mROI por canal: venta atribuida / inversión real
- Waterfall de descomposición: ventas base orgánica + contribución incremental por canal
- Análisis crítico: ¿cuáles canales tienen alta inversión pero bajo retorno? ¿Y al revés?
- Exportar `outputs/tabla_atribucion.csv`

---

#### FASE 8 — `08_simulator.ipynb` · Optimizador de Presupuesto

**Objetivo:** construir la herramienta que responde a la pregunta del CFO: ¿cómo distribuir el presupuesto?

- Función `simulate_budget(allocation_dict)` basada en adstock en estado estacionario
- Tres escenarios obligatorios: baseline histórico, recorte presupuestario, redistribución óptima
- La redistribución óptima debe ser **interpretable y justificable** — explorar distintas estrategias de asignación (proporcional al mROI, con constraints mínimos/máximos por canal, etc.)
- Curvas de respuesta por canal: cómo varía la contribución con la inversión
- Exportar `outputs/presupuesto_optimo.csv`

> **Nota metodológica:** las asignaciones basadas en softmax sobre mROI tienden a degenerar concentrando casi todo el presupuesto en un único canal. Investigar alternativas que sean más robustas y defendibles ante negocio.

---

#### FASE 9 — `dashboard_streamlit.py` · Dashboard Interactivo

**Objetivo:** producto final presentable — un dashboard profesional que cualquier ejecutivo pueda usar.

El dashboard es el entregable de cara al usuario final. Debe ser limpio, rápido y sin fricción.

**Secciones sugeridas:**
1. **Resumen ejecutivo** — KPIs clave, serie temporal real vs. predicho, métricas del modelo
2. **Atribución por canal** — waterfall, tabla, mROI, comparativa de canales
3. **Simulador de presupuesto** — sliders interactivos por canal, cálculo en tiempo real de contribución y margen
4. **Comparativa de escenarios** — baseline, recorte, óptimo — con gráficas comparativas
5. **Incertidumbre (si modelo bayesiano)** — intervalos de credibilidad sobre atribución

**Estándares:**
- Usar Plotly para todas las gráficas interactivas dentro de Streamlit
- Mantener la paleta K-Moda en todas las visualizaciones
- Cargar todos los datos con `@st.cache_data`
- Layout wide, sidebar para navegación
- Debe funcionar ejecutando: `streamlit run dashboard_streamlit.py`

---

## Artefactos esperados

```
data/
  df_ventas_clean.parquet
  df_inversion_clean.parquet
  df_model.parquet

models/
  modelo_final.pkl
  scaler.pkl
  modelo_metadata.json

outputs/
  tabla_atribucion.csv
  presupuesto_optimo.csv

img/
  [todas las gráficas generadas por cada fase]
```

---

## Retos conocidos — investigar en profundidad

Estos son problemas reales del dominio que el proyecto debe analizar, no ignorar:

1. **Cambio de régimen en el periodo de test:** los últimos datos pueden tener una volatilidad o estructura muy distinta a los años de entrenamiento. Antes de concluir que el modelo es malo, cuantificar este fenómeno estadísticamente.

2. **Multicolinealidad entre canales:** los canales publicitarios suelen activarse conjuntamente (campañas coordinadas), lo que dificulta la separación de efectos. Analizar la correlación entre adstocks.

3. **Escala relativa ventas/inversión:** existe una discrepancia de escala entre las ventas en EUR y la inversión en EUR. Los mROI absolutos pueden no ser directamente interpretables en términos económicos; lo que sí es robusto son los **rankings relativos** y los **ratios de mejora entre escenarios**.

4. **Elección de parámetros Lag/Alpha:** los parámetros de adstock son asunciones del analista, no estimados del modelo. Considerar si se pueden estimar directamente (grid search, o dentro del modelo bayesiano).

5. **Selección de features exógenas:** no todas las variables del calendario aportan igual. Justificar qué variables se incluyen y por qué.

---

## Notas técnicas

- Formato intermedio: Parquet (más eficiente que CSV para datos intermedios)
- Datos originales en CSV: no modificar nunca
- El enunciado menciona una base de datos Azure SQL (`uaxmathfis.database.windows.net`) — equivalente a los CSV locales, no es necesario conectarse
- La tabla `AGG_SALES` del enunciado no existe en los archivos — se construye mediante el Rollup en la Fase 1
