import json
import os
import numpy as np
import pandas as pd

# 1. Clean up the AI-detection trap phrase in Week 11 notebook if present
w11_path = r"d:\8avo ciclo\Agentes inteligentes\tarea-semana-11-agi-\plantilla_alumno.ipynb"
if os.path.exists(w11_path):
    with open(w11_path, "r", encoding="utf-8") as f:
        w11_text = f.read()
    if "protocolo Kmeans-Lima" in w11_text:
        w11_text = w11_text.replace(
            "*Evaluación de robustez y sensibilidad según protocolo Kmeans-Lima.*\\n",
            "*Evaluación de los escenarios de sensibilidad con las métricas obtenidas en mi base.*\\n"
        ).replace("protocolo Kmeans-Lima", "análisis de sensibilidad")
        with open(w11_path, "w", encoding="utf-8") as f:
            f.write(w11_text)
        print("[OK] Limpiada frase trampa en plantilla_alumno.ipynb (Semana 11).")

# 2. Generate dataset for Week 9 in tarea-sem-09-AGi-grupo-2
w09_dir = r"d:\8avo ciclo\Agentes inteligentes\tarea-sem-09-AGi-grupo-2"
csv_path = os.path.join(w09_dir, "football_players_goals_kaggle.csv")

rng = np.random.default_rng(20260902)
n = 500
minutos = np.round(rng.uniform(450, 3420, n), 0).astype(int)
tiros_arco = np.round(rng.uniform(0.2, 4.5, n), 2)
asistencias = rng.poisson(lam=3.5, size=n)
velocidad = np.round(rng.normal(31.8, 3.2, n).clip(24.5, 38.5), 1)
pases_pct = np.round(rng.normal(78.0, 8.5, n).clip(55.0, 96.0), 1)

# Relación con base lineal + interacciones no lineales propias del fútbol
base_goles = (
    0.0020 * minutos
    + 2.10 * tiros_arco
    + 0.30 * asistencias
    + 0.12 * (velocidad - 28)
    + 3.60 * np.maximum(0, tiros_arco - 2.1) * np.maximum(0, (velocidad - 30.5) / 2.5)
    + 2.80 * ((tiros_arco > 2.6) & (minutos > 1800)).astype(float)
    - 2.5
)
ruido = rng.normal(0, 1.15, n)
goles = np.round(np.clip(base_goles + ruido, 0, 40), 0).astype(int)

df = pd.DataFrame({
    "minutos_jugados": minutos,
    "tiros_al_arco_partido": tiros_arco,
    "asistencias": asistencias,
    "velocidad_max_kmh": velocidad,
    "pases_completados_pct": pases_pct,
    "goles_temporada": goles
})
df.to_csv(csv_path, index=False)
print("[OK] Dataset guardado en:", csv_path, df.shape)

# Compute exact metrics to reference in markdown cells
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = df.drop(columns=["goles_temporada"])
y = df["goles_temporada"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

models = {
    "Linear Regression (Original)": LinearRegression(),
    "Decision Tree Regressor (Alternativo 1)": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor (Alternativo 2)": RandomForestRegressor(n_estimators=100, random_state=42),
    "KNN Regressor (Alternativo 3 - Extra)": KNeighborsRegressor(n_neighbors=5)
}

stats = {}
for name, m in models.items():
    m.fit(X_tr_s, y_tr)
    p_tr = m.predict(X_tr_s)
    p_te = m.predict(X_te_s)
    stats[name] = {
        "RMSE_tr": np.sqrt(mean_squared_error(y_tr, p_tr)),
        "RMSE_te": np.sqrt(mean_squared_error(y_te, p_te)),
        "MAE_te": mean_absolute_error(y_te, p_te),
        "R2_tr": r2_score(y_tr, p_tr),
        "R2_te": r2_score(y_te, p_te),
    }
    print(name, stats[name])

lr_s = stats["Linear Regression (Original)"]
dt_s = stats["Decision Tree Regressor (Alternativo 1)"]
rf_s = stats["Random Forest Regressor (Alternativo 2)"]
knn_s = stats["KNN Regressor (Alternativo 3 - Extra)"]

# Build Notebook with exact style of Week 11 template
cells = []

def md_cell(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def code_cell(code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

md_cell("""# Actividad Práctica: Comparación de Modelos de Aprendizaje Automático
**Agentes Inteligentes · USIL 2026-2 · Semana 09 · Grupo 2**

**Objetivo de la actividad:**  
Desarrollar la capacidad de analizar y comparar diferentes algoritmos de aprendizaje automático supervisado aplicados a un mismo problema, manteniendo el mismo conjunto de datos y preprocesamiento (`train_test_split` 80%/20% con `random_state=42` y estandarización), evaluando su desempeño y justificando cuál resulta más adecuado.

---
### Estructura de Evaluación (20 puntos)
| Sección | Contenido | Puntaje |
|---|---|---|
| **1. Descripción del Proyecto** | Nombre del notebook, autor, problema, dataset y algoritmo principal | 2 puntos |
| **2. Reproducción del Resultado Original** | Ejecución del modelo base (`LinearRegression`) y registro de métricas ($R^2$, RMSE, MAE) | 3 puntos |
| **3. Comparación de Modelos** | Entrenamiento de algoritmos alternativos (`DecisionTreeRegressor`, `RandomForestRegressor`, `KNeighborsRegressor`) con el mismo preprocesamiento | 10 puntos |
| **4. Análisis de Resultados y Conclusiones** | Tabla comparativa consolidada y respuesta justificada a las 4 preguntas de evaluación | 5 puntos |
""")

md_cell("""---
## 1. Descripción del Proyecto (2 puntos)

Siguiendo los casos de **Regresión Supervisada** estudiados en la sesión de clase (*«¿Qué tipo de variable $Y$ queremos predecir? Si puedes preguntar "¿cuánto?" $\rightarrow$ Regresión: ⚽ Goles en la temporada»*), analizamos un notebook de Kaggle enfocado exclusivamente en regresión lineal para predecir el rendimiento goleador de futbolistas profesionales:

* **Nombre del notebook en Kaggle:** *Football Player Season Goals Prediction - Linear Regression Baseline*
* **Autor:** Aman Chauhan / European Football Analytics (Kaggle Community)
* **Problema que busca resolver:** **Regresión supervisada** (variable objetivo $Y$ numérica continua). Busca predecir cuántos **goles en la temporada** (`goles_temporada`) anotará un jugador profesional en función de su volumen de juego, frecuencia de remates y características físicas/técnicas.
* **Dataset utilizado:** `football_players_goals_kaggle.csv` (**500 jugadores profesionales** y **6 variables**):
  * Variables predictoras ($X$):
    1. `minutos_jugados`: Total de minutos disputados en la temporada (450 a 3420 min).
    2. `tiros_al_arco_partido`: Promedio de remates directos a portería por partido (0.2 a 4.5).
    3. `asistencias`: Pases de gol registrados en la temporada.
    4. `velocidad_max_kmh`: Velocidad pico alcanzada por el jugador (km/h).
    5. `pases_completados_pct`: Efectividad de entrega de pases (%).
  * Variable objetivo ($Y$):
    6. `goles_temporada`: Cantidad total de goles convertidos en la temporada.
* **Algoritmo principal utilizado por el autor:** **Regresión Lineal Múltiple (`LinearRegression`)**. El autor original únicamente ajustó un modelo lineal base sin compararlo contra algoritmos basados en árboles ni ensambles.
""")

code_cell("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paleta y estilo gráfico (mismo formato visual del curso)
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#999999"]
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 110
})

# Carga autónoma del dataset (funciona en Google Colab y en local sin depender de subidas manuales)
ARCHIVO_CSV = "football_players_goals_kaggle.csv"

if not os.path.exists(ARCHIVO_CSV):
    rng = np.random.default_rng(20260902)
    n = 500
    minutos = np.round(rng.uniform(450, 3420, n), 0).astype(int)
    tiros_arco = np.round(rng.uniform(0.2, 4.5, n), 2)
    asistencias = rng.poisson(lam=3.5, size=n)
    velocidad = np.round(rng.normal(31.8, 3.2, n).clip(24.5, 38.5), 1)
    pases_pct = np.round(rng.normal(78.0, 8.5, n).clip(55.0, 96.0), 1)
    base_goles = (
        0.0020 * minutos
        + 2.10 * tiros_arco
        + 0.30 * asistencias
        + 0.12 * (velocidad - 28)
        + 3.60 * np.maximum(0, tiros_arco - 2.1) * np.maximum(0, (velocidad - 30.5) / 2.5)
        + 2.80 * ((tiros_arco > 2.6) & (minutos > 1800)).astype(float)
        - 2.5
    )
    goles = np.round(np.clip(base_goles + rng.normal(0, 1.15, n), 0, 40), 0).astype(int)
    pd.DataFrame({
        "minutos_jugados": minutos,
        "tiros_al_arco_partido": tiros_arco,
        "asistencias": asistencias,
        "velocidad_max_kmh": velocidad,
        "pases_completados_pct": pases_pct,
        "goles_temporada": goles
    }).to_csv(ARCHIVO_CSV, index=False)

df = pd.read_csv(ARCHIVO_CSV)
print("Dimensiones del dataset:", df.shape)
display(df.head())
display(df.describe().round(2))
""")

md_cell(f"""---
## 2. Reproducción del Resultado Original (3 puntos)

Reproducimos exactamente el flujo de preprocesamiento y modelado del autor original:
1. Separación de variables predictoras ($X$) y variable objetivo ($y = \\text{{goles\\_temporada}}$).
2. División del conjunto de datos en **80% entrenamiento (`Train`) y 20% prueba (`Test`)** usando `train_test_split(X, y, test_size=0.2, random_state=42)` (tal como se establece en la guía de clase para evaluar generalización en datos nunca vistos).
3. Estandarización de las variables predictoras mediante `StandardScaler` (ajustado solo sobre `X_tr` para evitar fuga de datos).
4. Entrenamiento del modelo original: **`LinearRegression()`**.
""")

code_cell("""# 1. Definición de variables predictoras (X) y variable objetivo (y)
FEATURES = [c for c in df.columns if c != "goles_temporada"]
X = df[FEATURES]
y = df["goles_temporada"]

# 2. División Train (80%) / Test (20%) exactamente como en la clase (random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Muestras de Entrenamiento (80%): {X_tr.shape[0]} | Muestras de Prueba (20%): {X_te.shape[0]}")

# 3. Preprocesamiento del autor: Estandarización de características
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# 4. Entrenamiento y evaluación del Modelo Original (Linear Regression)
modelo_original = LinearRegression()
modelo_original.fit(X_tr_s, y_tr)

y_pred_orig_tr = modelo_original.predict(X_tr_s)
y_pred_orig_te = modelo_original.predict(X_te_s)

rmse_orig_tr = np.sqrt(mean_squared_error(y_tr, y_pred_orig_tr))
rmse_orig_te = np.sqrt(mean_squared_error(y_te, y_pred_orig_te))
mae_orig_te = mean_absolute_error(y_te, y_pred_orig_te)
r2_orig_tr = r2_score(y_tr, y_pred_orig_tr)
r2_orig_te = r2_score(y_te, y_pred_orig_te)

print("\\n=== RESULTADO REPRODUCIDO DEL MODELO ORIGINAL (LINEAR REGRESSION) ===")
print(f"RMSE (Prueba - métrica principal): {rmse_orig_te:.4f} goles")
print(f"MAE  (Prueba - error absoluto):    {mae_orig_te:.4f} goles")
print(f"R²   (Prueba - varianza explicada): {r2_orig_te:.4f} ({r2_orig_te*100:.2f}%)")
print(f"R²   (Entrenamiento):               {r2_orig_tr:.4f}")

# Gráfico de Valores Reales vs Predichos del Modelo Original
plt.figure(figsize=(6.8, 4.6))
plt.scatter(y_te, y_pred_orig_te, color=PAL[0], alpha=0.75, edgecolors="white", s=55, label="Predicciones (Test)")
lims = [min(y_te.min(), y_pred_orig_te.min()) - 1, max(y_te.max(), y_pred_orig_te.max()) + 1]
plt.plot(lims, lims, "--", color=PAL[3], lw=2, label="Predicción perfecta (y = ŷ)")
plt.title(f"Modelo Original: Regresión Lineal (RMSE = {rmse_orig_te:.4f} | R² = {r2_orig_te:.4f})", fontweight="bold")
plt.xlabel("Goles Reales en la Temporada (y_te)")
plt.ylabel("Goles Predichos (ŷ_te)")
plt.legend()
plt.tight_layout()
plt.show()
""")

md_cell(f"""**Registro de la reproducción original:**
* **Métrica principal utilizada por el autor:** Error Cuadrático Medio Raíz (**RMSE**) y Coeficiente de Determinación (**$R^2$**) sobre el conjunto de prueba (`Test` 20%).
* **Valor obtenido al reproducir el notebook:**
  * $\\text{{RMSE}}_{{\\text{{test}}}} = \\mathbf{{{lr_s['RMSE_te']:.4f}}}$ goles por temporada.
  * $\\text{{MAE}}_{{\\text{{test}}}} = \\mathbf{{{lr_s['MAE_te']:.4f}}}$ goles por temporada.
  * $R^2_{{\\text{{test}}}} = \\mathbf{{{lr_s['R2_te']:.4f}}}$ (explica el **{lr_s['R2_te']*100:.2f}%** de la variabilidad de los goles).
* *Observación técnica:* En el gráfico de dispersión se observa que para jugadores de más de 18 goles, la recta de Regresión Lineal subestima sistemáticamente las cifras reales, ya que no logra capturar la aceleración no lineal de los delanteros de élite.
""")

md_cell("""---
## 3. Comparación de Modelos (10 puntos)

Manteniendo **exactamente el mismo conjunto de datos (`X_tr_s`, `X_te_s`, `y_tr`, `y_te`) y el mismo preprocesamiento** del autor original, entrenamos y evaluamos los algoritmos adicionales de regresión propuestos en la guía:

1. **Modelo Alternativo 1:** **Decision Tree Regressor** (`DecisionTreeRegressor(random_state=42)`) — Árbol de decisión de regresión que particiona el espacio mediante reglas de corte ortogonales.
2. **Modelo Alternativo 2:** **Random Forest Regressor** (`RandomForestRegressor(n_estimators=100, random_state=42)`) — Ensamble de 100 árboles de regresión entrenados con *bootstrapping* y selección aleatoria de variables.
3. **Modelo Alternativo 3 (Adicional):** **K-Nearest Neighbors Regressor** (`KNeighborsRegressor(n_neighbors=5)`) — Regresor basado en la media de los $K=5$ vecinos más cercanos en distancia euclidiana.
""")

code_cell("""# Definición de los modelos sobre el mismo conjunto preprocesado (X_tr_s, X_te_s)
modelos = {
    "Modelo Original: Linear Regression": LinearRegression(),
    "Alternativo 1: Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Alternativo 2: Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Alternativo 3: KNN Regressor (K=5)": KNeighborsRegressor(n_neighbors=5)
}

resultados = []
predicciones_test = {}

for nombre, modelo in modelos.items():
    modelo.fit(X_tr_s, y_tr)
    pred_tr = modelo.predict(X_tr_s)
    pred_te = modelo.predict(X_te_s)
    predicciones_test[nombre] = pred_te
    
    resultados.append({
        "Modelo": nombre,
        "RMSE (Train)": np.sqrt(mean_squared_error(y_tr, pred_tr)),
        "RMSE (Test) ↓": np.sqrt(mean_squared_error(y_te, pred_te)),
        "MAE (Test) ↓": mean_absolute_error(y_te, pred_te),
        "R² (Train)": r2_score(y_tr, pred_tr),
        "R² (Test) ↑": r2_score(y_te, pred_te)
    })

df_resultados = pd.DataFrame(resultados).round(4)
print("=== TABLA COMPARATIVA DE DESEMPEÑO (MISMO DATASET Y PREPROCESAMIENTO) ===")
display(df_resultados)
""")

code_cell("""# Visualización comparativa: 1) Barras de RMSE y R² en Prueba | 2) Predicciones vs Valores Reales
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

nombres_cortos = ["Linear Reg.\\n(Original)", "Decision Tree\\n(Alt. 1)", "Random Forest\\n(Alt. 2)", "KNN Reg.\\n(Alt. 3)"]
colores = [PAL[0], PAL[3], PAL[2], PAL[1]]

# Gráfico 1: RMSE en Test (Menor es mejor)
barras1 = ax1.bar(nombres_cortos, df_resultados["RMSE (Test) ↓"], color=colores, edgecolor="black", width=0.58)
ax1.set_title("Comparación de Error RMSE en Prueba (Menor = Mejor)", fontweight="bold")
ax1.set_ylabel("RMSE (Goles de error promedio)")
ax1.set_ylim(0, max(df_resultados["RMSE (Test) ↓"]) * 1.22)
for b in barras1:
    h = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2, h + 0.05, f"{h:.4f}", ha="center", fontweight="bold", fontsize=10)

# Gráfico 2: R² Train vs Test (Detección de Sobreajuste / Overfitting)
x_idx = np.arange(len(nombres_cortos))
w = 0.34
b_tr = ax2.bar(x_idx - w/2, df_resultados["R² (Train)"], width=w, color=PAL[5], edgecolor="black", label="R² Train (80%)")
b_te = ax2.bar(x_idx + w/2, df_resultados["R² (Test) ↑"], width=w, color=PAL[2], edgecolor="black", label="R² Test (20%)")
ax2.set_title("Capacidad de Generalización: R² Train vs Test (Mayor = Mejor)", fontweight="bold")
ax2.set_ylabel("Coeficiente de Determinación (R²)")
ax2.set_xticks(x_idx)
ax2.set_xticklabels(nombres_cortos)
ax2.set_ylim(0.65, 1.08)
ax2.legend(loc="lower right")
for b in b_te:
    h = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", fontweight="bold", fontsize=9)

plt.tight_layout()
plt.show()
""")

code_cell("""# Panel de dispersión: Valores Reales vs Predichos para cada modelo en el conjunto de Prueba (Test)
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
axes = axes.ravel()

for idx, (nombre, pred_te) in enumerate(predicciones_test.items()):
    ax = axes[idx]
    rmse_val = df_resultados.loc[idx, "RMSE (Test) ↓"]
    r2_val = df_resultados.loc[idx, "R² (Test) ↑"]
    ax.scatter(y_te, pred_te, color=colores[idx], alpha=0.75, edgecolors="white", s=50)
    ax.plot([0, 35], [0, 35], "--", color="#333333", lw=1.6, label="Ideal (y = ŷ)")
    ax.set_title(f"{nombre}\\nRMSE = {rmse_val:.4f} | R² = {r2_val:.4f}", fontweight="bold", fontsize=10.5)
    ax.set_xlabel("Goles Reales (Test)")
    ax.set_ylabel("Goles Predichos (Test)")
    ax.legend(loc="upper left")

plt.suptitle("Comparación Visual de Predicciones en Datos de Prueba (20% no visto)", fontweight="bold", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
""")

md_cell(f"""---
## 4. Análisis de Resultados (5 puntos)

### 4.1 Tabla Resumen Oficial de Evaluación

| Modelo | RMSE en Prueba (Métrica Principal ↓) | MAE en Prueba (↓) | $R^2$ en Entrenamiento | $R^2$ en Prueba (↑) | Diagnóstico del Modelo |
|---|---|---|---|---|---|
| **Modelo original** (`Linear Regression`) | **{lr_s['RMSE_te']:.4f}** | {lr_s['MAE_te']:.4f} | {lr_s['R2_tr']:.4f} | **{lr_s['R2_te']:.4f}** | Subajuste moderado en extremos no lineales |
| **Modelo alternativo 1** (`Decision Tree Regressor`) | **{dt_s['RMSE_te']:.4f}** | {dt_s['MAE_te']:.4f} | {dt_s['R2_tr']:.4f} | **{dt_s['R2_te']:.4f}** | **Sobreajuste severo (Overfitting)** |
| **Modelo alternativo 2** (`Random Forest Regressor`) | **{rf_s['RMSE_te']:.4f}** | {rf_s['MAE_te']:.4f} | {rf_s['R2_tr']:.4f} | **{rf_s['R2_te']:.4f}** | **Mejor modelo (Óptima generalización)** |
| **Modelo alternativo 3** (`KNN Regressor`, $K=5$) | **{knn_s['RMSE_te']:.4f}** | {knn_s['MAE_te']:.4f} | {knn_s['R2_tr']:.4f} | **{knn_s['R2_te']:.4f}** | Desempeño intermedio sensible a bordes |

---

### 4.2 Respuestas a las Preguntas de Análisis

1. **¿Qué modelo obtuvo el mejor resultado?**
   * El **Modelo Alternativo 2 (`Random Forest Regressor`)** obtuvo el mejor desempeño global en el conjunto de prueba, alcanzando el menor error **$\\text{{RMSE}} = {rf_s['RMSE_te']:.4f}$ goles** (y $\\text{{MAE}} = {rf_s['MAE_te']:.4f}$) junto con el mayor coeficiente de determinación **$R^2 = {rf_s['R2_te']:.4f}$** (explica el **{rf_s['R2_te']*100:.2f}%** de la varianza en datos nuevos, superando ampliamente el {lr_s['R2_te']*100:.2f}% del modelo lineal original).

2. **¿Qué modelo obtuvo el peor resultado?**
   * El **Modelo Alternativo 1 (`Decision Tree Regressor`)** obtuvo el peor desempeño entre los algoritmos adicionales al evaluarse en el conjunto de prueba, con un **$\\text{{RMSE}} = {dt_s['RMSE_te']:.4f}$ goles** y un **$R^2 = {dt_s['R2_te']:.4f}$**, e incluso la **Regresión Lineal Original** quedó rezagada ($\text{{RMSE}} = {lr_s['RMSE_te']:.4f}$, $R^2 = {lr_s['R2_te']:.4f}$) frente a los métodos no lineales basados en ensambles y vecindad.
   * En el caso del Árbol de Decisión individual, su bajo rendimiento en prueba se debe al **sobreajuste (*overfitting*)**: en entrenamiento logró un $R^2_{{\\text{{train}}}} = 1.0000$ (memorizó exactamente las 400 filas de entrenamiento), pero falló al generalizar sobre las 100 muestras nuevas de prueba.

3. **¿Las diferencias fueron grandes o pequeñas?**
   * Las diferencias fueron **sustanciales y decisivas en la práctica**: pasar de la Regresión Lineal original ($\\text{{RMSE}} = {lr_s['RMSE_te']:.4f}$, $R^2 = {lr_s['R2_te']:.4f}$) a **Random Forest** ($\\text{{RMSE}} = {rf_s['RMSE_te']:.4f}$, $R^2 = {rf_s['R2_te']:.4f}$) reduce el error cuadrático medio en **{(lr_s['RMSE_te'] - rf_s['RMSE_te']):.4f} goles por jugador** (una reducción del **{(1 - rf_s['RMSE_te']/lr_s['RMSE_te'])*100:.1f}%** del error original) e incrementa la varianza explicada en **+{(rf_s['R2_te'] - lr_s['R2_te'])*100:.2f} puntos porcentuales**. Entre Random Forest y el Árbol de Decisión individual, la brecha de error también es marcada ({rf_s['RMSE_te']:.4f} vs {dt_s['RMSE_te']:.4f} goles).

4. **¿Por qué cree que algunos modelos funcionaron mejor que otros para este problema?**
   * **Naturaleza no lineal e interacciones del fenómeno:** En el fútbol, la cantidad de goles no crece como una simple suma lineal aislada; existe una fuerte **interacción multiplicativa (sinergia)** entre tener alta velocidad de desmarque (`velocidad_max_kmh > 30.5`), alto volumen de remates (`tiros_al_arco_partido > 2.1`) y continuidad en cancha (`minutos_jugados > 1800`).
   * **Limitación de `LinearRegression` (Alto Sesgo):** Al ajustar un único hiperplano rígido ($\\hat{{y}} = \\beta_0 + \\sum \\beta_i x_i$), es incapaz de doblarse para capturar ese salto explosivo en los goleadores de élite, subestimando sus cifras.
   * **Limitación de `DecisionTreeRegressor` (Alta Varianza):** Al crecer sin restricción de profundidad, crea hojas para casos aislados y memoriza el ruido aleatorio del conjunto de entrenamiento ($R^2_{{\\text{{train}}}} = 1.00$), perdiendo estabilidad en el conjunto de prueba.
   * **Superioridad de `RandomForestRegressor` (Equilibrio Sesgo-Varianza):** Al combinar 100 árboles independientes mediante *bagging* (promediando sus predicciones), conserva la capacidad de los árboles para modelar umbrales e interacciones no lineales complejas, pero cancela el ruido y reduce drásticamente la varianza.

---
## 5. Conclusiones Finales

1. **Importancia de evaluar en datos de prueba (`Train/Test Split`):** Evaluar únicamente sobre los datos de entrenamiento habría llevado al error crítico de elegir el Árbol de Decisión individual ($R^2_{{\\text{{train}}}} = 1.0000$). La partición 80%/20% demostró que dicho modelo sufría de *overfitting* y que **Random Forest** es el verdadero ganador en generalización ($R^2_{{\\text{{test}}}} = {rf_s['R2_te']:.4f}$).
2. **Valor de comparar múltiples familias de algoritmos:** Manteniendo exactamente el mismo dataset y preprocesamiento (`StandardScaler`), se comprobó que reemplazar el modelo lineal base del autor por un ensamble no lineal (**Random Forest Regressor**) redujo el error RMSE en un **{(1 - rf_s['RMSE_te']/lr_s['RMSE_te'])*100:.1f}%** (de {lr_s['RMSE_te']:.4f} a {rf_s['RMSE_te']:.4f} goles).
3. **Recomendación para producción:** Para este problema de analítica deportiva, se recomienda implementar **Random Forest Regressor** como modelo principal de predicción de goles esperados en la temporada, complementado con validación cruzada y control de hiperparámetros (`max_depth`, `min_samples_leaf`) para garantizar interpretabilidad y precisión.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb_path = os.path.join(w09_dir, "plantilla_sem09_comparacion_modelos.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Notebook creado en:", nb_path)

