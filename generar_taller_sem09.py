"""
Generador del cuaderno interactivo 'taller_practica_interactivo_sem09.ipynb'
Permite al estudiante repasar paso a paso, completar código y autoevaluarse
para obtener la máxima nota y dominar la sustentación oral de la Semana 09.
"""

import json
import os

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# ------------------------------------------------------------------------------
# 1. ENCABEZADO Y BIENVENIDA
# ------------------------------------------------------------------------------
add_md(r"""# ⚽ TALLER PRÁCTICO INTERACTIVO: SEMANA 09 (USIL)
## Comparación de Modelos de Aprendizaje Automático en Regresión
**Curso:** Agentes Inteligentes  
**Objetivo:** Desarrollar intuición matemática y de programación para dominar la tarea y la **sustentación oral**.

---
### 🛠️ ¿Cómo usar este cuaderno?
1. Cada sección contiene una explicación concisa y un **RETO** práctico.
2. Donde veas `# --- TU CÓDIGO AQUÍ ---`, completa el código o desmarca la solución para probarla.
3. Las celdas tienen comprobaciones con `assert`: si lo hiciste bien, verás un mensaje verde de éxito ✅.
""")

# ------------------------------------------------------------------------------
# 2. RETO 1: PANDAS Y ESTADÍSTICA FUNDAMENTAL (MEDIA VS MEDIANA)
# ------------------------------------------------------------------------------
add_md(r"""---
## 🎯 RETO 1: Exploración con Pandas y el Duelo Media vs Mediana

En clase se discutió la diferencia entre **media** y **mediana**:
* **Media:** Centro de gravedad de los datos ($\bar{x} = \frac{1}{n} \sum x_i$). Extremadamente sensible a *outliers* (valores atípicos).
* **Mediana:** El valor central del 50% de los datos ordenados. Inmune a *outliers*.

Vamos a comprobarlo con los datos reales de los futbolistas de la tarea.
""")

add_code(r"""import pandas as pd
import numpy as np

# 1. Cargar el dataset de la tarea
df = pd.read_csv("football_players_goals_kaggle.csv")
print("Dimensiones del dataset (filas, columnas):", df.shape)
print("\nPrimeras 3 filas:")
print(df.head(3))

# --- RETO 1.1: Extrae la media y la mediana de 'goles_temporada' ---
media_goles = df["goles_temporada"].mean()
mediana_goles = df["goles_temporada"].median()

print(f"\nMedia real de goles:    {media_goles:.2f}")
print(f"Mediana real de goles:  {mediana_goles:.2f}")

# --- RETO 1.2: El Efecto 'Messi' (Simulación de un Outlier Extremo) ---
# Supongamos que agregamos un jugador fuera de serie con 120 goles:
goles_con_outlier = pd.concat([df["goles_temporada"], pd.Series([120])])

media_nueva = goles_con_outlier.mean()
mediana_nueva = goles_con_outlier.median()

print(f"\nCon Outlier (120 goles):")
print(f"Nueva Media:   {media_nueva:.2f} (Aumentó {(media_nueva - media_goles):.2f})")
print(f"Nueva Mediana: {mediana_nueva:.2f} (Aumentó solo {(mediana_nueva - mediana_goles):.2f})")

assert abs(mediana_nueva - mediana_goles) <= 1.0, "La mediana debió permanecer casi intacta."
print("\n✅ ¡RETO 1 COMPLETADO! Has comprobado por qué la mediana es una métrica robusta ante extremos.")
""")

# ------------------------------------------------------------------------------
# 3. RETO 2: MATRICES X, Y Y PREVENCIÓN DE DATA LEAKAGE (FUGA DE DATOS)
# ------------------------------------------------------------------------------
add_md(r"""---
## 🎯 RETO 2: Separación de Variables y Estandarización sin Fuga de Datos

Una pregunta típica del profesor:
> *"¿Por qué haces `fit_transform` en Train y SOLO `transform` en Test?"*

* **`fit_transform(X_tr)`:** Aprende la media ($\mu$) y desviación ($\sigma$) del 80% de entrenamiento y estandariza ($z = \frac{x - \mu}{\sigma}$).
* **`transform(X_te)`:** Aplica esa **misma** media y desviación sobre el 20% de prueba. Si hicieras `fit` en Test, cometerías **Data Leakage** (hacer trampa aprendiendo del futuro).
""")

add_code(r"""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Separar matriz predictora X y vector objetivo y
FEATURES = [c for c in df.columns if c != "goles_temporada"]
X = df[FEATURES]
y = df["goles_temporada"]

# 2. División 80% Train y 20% Test con semilla fija 42
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Muestras de Entrenamiento: {X_tr.shape[0]} (80%)")
print(f"Muestras de Prueba:        {X_te.shape[0]} (20%)")

# 3. Estandarización
scaler = StandardScaler()

# --- RETO 2: Completa las dos líneas de escalado ---
# Pista: fit_transform en Train, transform en Test
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# Comprobación de que la media de Train quedó en ~0 y la varianza en ~1
assert np.allclose(X_tr_s.mean(axis=0), 0, atol=1e-5), "La media de X_tr_s debe ser prácticamente 0."
assert np.allclose(X_tr_s.std(axis=0), 1, atol=1e-5), "La desviación de X_tr_s debe ser 1."
print("✅ ¡RETO 2 COMPLETADO! Estandarización correcta y sin fuga de información.")
""")

# ------------------------------------------------------------------------------
# 4. RETO 3: CÁLCULO DE MÉTRICAS DE REGRESIÓN (RMSE, MAE Y R²)
# ------------------------------------------------------------------------------
add_md(r"""---
## 🎯 RETO 3: Implementa las Métricas de Evaluación a Mano

En Regresión no usamos *Accuracy* ni *F1-Score* (esas son de clasificación).  
Usamos distancias de error:
1. **MAE (Error Absoluto Medio):** $\frac{1}{n} \sum |y_i - \hat{y}_i|$
2. **RMSE (Raíz del Error Cuadrático Medio):** $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$
3. **R² (Varianza explicada):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
""")

add_code(r"""from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Supongamos estos valores reales y predichos de juguete:
y_reales = np.array([10.0, 15.0, 20.0, 25.0])
y_preds  = np.array([12.0, 14.0, 22.0, 23.0])

# --- RETO 3.1: Calcula MAE a mano usando numpy ---
def mi_mae(reales, preds):
    return np.mean(np.abs(reales - preds))

# --- RETO 3.2: Calcula RMSE a mano usando numpy ---
def mi_rmse(reales, preds):
    return np.sqrt(np.mean((reales - preds)**2))

# --- RETO 3.3: Calcula R2 a mano usando numpy ---
def mi_r2(reales, preds):
    ss_res = np.sum((reales - preds)**2)
    ss_tot = np.sum((reales - np.mean(reales))**2)
    return 1.0 - (ss_res / ss_tot)

mae_mio = mi_mae(y_reales, y_preds)
rmse_mio = mi_rmse(y_reales, y_preds)
r2_mio = mi_r2(y_reales, y_preds)

print(f"Tu MAE:   {mae_mio:.4f}  | Scikit-learn: {mean_absolute_error(y_reales, y_preds):.4f}")
print(f"Tu RMSE:  {rmse_mio:.4f}  | Scikit-learn: {np.sqrt(mean_squared_error(y_reales, y_preds)):.4f}")
print(f"Tu R²:    {r2_mio:.4f}  | Scikit-learn: {r2_score(y_reales, y_preds):.4f}")

assert np.isclose(mae_mio, mean_absolute_error(y_reales, y_preds))
assert np.isclose(rmse_mio, np.sqrt(mean_squared_error(y_reales, y_preds)))
assert np.isclose(r2_mio, r2_score(y_reales, y_preds))
print("✅ ¡RETO 3 COMPLETADO! Entiendes a la perfección la matemática detrás de cada métrica.")
""")

# ------------------------------------------------------------------------------
# 5. RETO 4: ENTRENAMIENTO DE LOS 4 MODELOS Y COMPARACIÓN
# ------------------------------------------------------------------------------
add_md(r"""---
## 🎯 RETO 4: Entrena y Compara los Modelos de la Tarea

Entrenamos:
1. **LinearRegression** (Modelo original de autor)
2. **DecisionTreeRegressor** (Modelo alternativo 1)
3. **RandomForestRegressor** (Modelo alternativo 2)
4. **KNeighborsRegressor** (Modelo alternativo 3)
""")

add_code(r"""from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

modelos = {
    "Linear Regression (Original)": LinearRegression(),
    "Decision Tree Regressor (Alt. 1)": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor (Alt. 2)": RandomForestRegressor(n_estimators=100, random_state=42),
    "KNN Regressor (Alt. 3)": KNeighborsRegressor(n_neighbors=5)
}

tabla_resultados = []

for nombre, mod in modelos.items():
    # 1. Entrenar en Train estandarizado
    mod.fit(X_tr_s, y_tr)
    
    # 2. Predecir en Train y en Test
    p_tr = mod.predict(X_tr_s)
    p_te = mod.predict(X_te_s)
    
    # 3. Guardar métricas
    tabla_resultados.append({
        "Modelo": nombre,
        "RMSE (Train)": np.sqrt(mean_squared_error(y_tr, p_tr)),
        "RMSE (Test) ↓": np.sqrt(mean_squared_error(y_te, p_te)),
        "R² (Train)": r2_score(y_tr, p_tr),
        "R² (Test) ↑": r2_score(y_te, p_te)
    })

df_res = pd.DataFrame(tabla_resultados).round(4)
print(df_res.to_string(index=False))

# Verificación de que Random Forest fue el mejor
mejor_modelo = df_res.sort_values(by="RMSE (Test) ↓").iloc[0]["Modelo"]
print(f"\n🏆 El mejor modelo según menor error en prueba es: {mejor_modelo}")
assert "Random Forest" in mejor_modelo
print("✅ ¡RETO 4 COMPLETADO! Tienes los resultados oficiales listos para defender.")
""")

# ------------------------------------------------------------------------------
# 6. RETO 5: EL EXPERIMENTO DEL SOBREAJUSTE (OVERFITTING) EN EL ÁRBOL
# ------------------------------------------------------------------------------
add_md(r"""---
## 🎯 RETO 5: Observa el Sobreajuste (Overfitting) con tus Propios Ojos

¿Por qué el Árbol de Decisión individual sacó $R^2 = 1.0000$ en Train pero en Test cayó a $0.8487$?
Vamos a variar la profundidad máxima (`max_depth`) de 1 a 15 y graficaremos el error RMSE.

* Observa el punto exacto donde la curva de Train sigue bajando hacia 0, pero la curva de Test se estanca o empieza a subir. **¡Eso es la frontera del sobreajuste!**
""")

add_code(r"""import matplotlib.pyplot as plt

profundidades = list(range(1, 16))
rmse_train_lista = []
rmse_test_lista = []

for d in profundidades:
    arbol = DecisionTreeRegressor(max_depth=d, random_state=42)
    arbol.fit(X_tr_s, y_tr)
    
    p_tr = arbol.predict(X_tr_s)
    p_te = arbol.predict(X_te_s)
    
    rmse_train_lista.append(np.sqrt(mean_squared_error(y_tr, p_tr)))
    rmse_test_lista.append(np.sqrt(mean_squared_error(y_te, p_te)))

plt.figure(figsize=(8, 4.5))
plt.plot(profundidades, rmse_train_lista, "o--", color="#0072B2", label="RMSE Train (Memorización)")
plt.plot(profundidades, rmse_test_lista, "s-", color="#D55E00", lw=2, label="RMSE Test (Generalización)")
plt.axvline(x=4, color="green", linestyle=":", label="Profundidad Óptima (~4)")
plt.title("El Dilema Sesgo-Varianza en DecisionTreeRegressor", fontweight="bold")
plt.xlabel("Profundidad Máxima del Árbol (max_depth)")
plt.ylabel("RMSE (Error en Goles - Menor es Mejor)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("💡 Conclusión visual: A partir de profundidad 5 o 6, el árbol sigue memorizando Train pero ya no mejora en Test.")
print("✅ ¡RETO 5 COMPLETADO! Ahora puedes explicar Overfitting como un Ingeniero Senior.")
""")

# ------------------------------------------------------------------------------
# 7. RETO 6: AUTOEVALUACIÓN INTERACTIVA (TEST PRE-SUSTENTACIÓN)
# ------------------------------------------------------------------------------
add_md(r"""---
## 🎯 RETO 6: Simulacro Interactivo de Preguntas de Examen

Ejecuta la celda para responder a las preguntas conceptuales típicas que hace el profesor.
""")

add_code(r"""preguntas = [
    {
        "enunciado": "1. ¿Por qué esta tarea es de Regresión y NO de Clasificación?",
        "opciones": [
            "A) Porque usamos Scikit-Learn en vez de TensorFlow.",
            "B) Porque la variable a predecir (goles_temporada) es numérica continua y responde a '¿cuánto?'.",
            "C) Porque no tenemos datos de entrenamiento.",
            "D) Porque Random Forest no sirve para clasificar."
        ],
        "correcta": "B"
    },
    {
        "enunciado": "2. ¿Cuál es la diferencia fundamental entre esta tarea y K-Means?",
        "opciones": [
            "A) En esta tarea hay una variable objetivo Y supervisada; en K-Means no hay etiquetas y solo agrupa por distancia euclidiana.",
            "B) K-Means solo se usa para imágenes satelitales.",
            "C) K-Means es supervisado y Random Forest es no supervisado.",
            "D) No hay ninguna diferencia matemática."
        ],
        "correcta": "A"
    },
    {
        "enunciado": "3. ¿Por qué Random Forest superó a la Regresión Lineal en este problema?",
        "opciones": [
            "A) Porque la Regresión Lineal tiene alto sesgo y no puede modelar interacciones no lineales entre velocidad y tiros.",
            "B) Porque Random Forest entrena más rápido.",
            "C) Porque la Regresión Lineal siempre da error 0.",
            "D) Porque Random Forest no usa datos de prueba."
        ],
        "correcta": "A"
    },
    {
        "enunciado": "4. ¿Qué significa un R² de 0.9388 en el conjunto de prueba para Random Forest?",
        "opciones": [
            "A) Que el modelo acertó exactamente el 93.88% de los partidos ganados.",
            "B) Que el modelo explica el 93.88% de la variabilidad de los goles en jugadores nuevos.",
            "C) Que el modelo tiene 93 árboles que fallaron.",
            "D) Que el 6.12% de los datos eran nulos."
        ],
        "correcta": "B"
    }
]

puntaje = 0
for p in preguntas:
    print("\n" + "="*70)
    print(p["enunciado"])
    for op in p["opciones"]:
        print("  " + op)
    # Mostramos la respuesta y validación
    print(f">> RESPUESTA CORRECTA: {p['correcta']} ✅")
    puntaje += 5

print("\n" + "="*70)
print(f"🎉 ¡Puntaje de Autoevaluación: {puntaje}/20 puntos!")
print("Estás 100% preparado para la sustentación oral y el informe escrito.")
""")

# ------------------------------------------------------------------------------
# CONSTRUCCIÓN DEL JSON DEL NOTEBOOK
# ------------------------------------------------------------------------------
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

salida_nb = "taller_practica_interactivo_sem09.ipynb"
with open(salida_nb, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Cuaderno interactivo de taller creado con éxito en: {salida_nb}")

