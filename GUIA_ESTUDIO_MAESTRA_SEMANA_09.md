# 🎓 GUÍA DE ESTUDIO MAESTRA: COMPARACIÓN DE MODELOS DE APRENDIZAJE AUTOMÁTICO
**Curso:** Agentes Inteligentes — Unidad 3 (Semana 09)  
**Institución:** Universidad San Ignacio de Loyola (USIL)  
**Tema:** Comparación de Algoritmos Supervisados de Regresión (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `KNeighborsRegressor`)  
**Nivel de Explicación:** Ingeniero Senior en Inteligencia Artificial y Machine Learning  

---

## 📑 ÍNDICE DE CONTENIDOS
1. [El Gran Mapa Mental de la IA: ¿Dónde encaja la Semana 9?](#1-el-gran-mapa-mental-de-la-ia)
2. [Conexión con Agentes Inteligentes: Ciclo Percepción-Razonamiento-Acción](#2-conexión-con-agentes-inteligentes)
3. [Estadística Fundamental: Media, Mediana y el Dilema Sesgo-Varianza](#3-estadística-fundamental)
4. [Métricas de Evaluación de Regresión: RMSE, MAE y R² Explicados al Detalle](#4-métricas-de-evaluación-de-regresión)
5. [Pandas y Preprocesamiento de Datos: Análisis Línea por Línea](#5-pandas-y-preprocesamiento-de-datos)
6. [Los 4 Algoritmos Evaluados: Cómo funcionan bajo el capó](#6-los-4-algoritmos-evaluados)
7. [Análisis de los Resultados Experimentales (Datos Reales de la Tarea)](#7-análisis-de-los-resultados-experimentales)
8. [Desglose Completo del Código de la Tarea](#8-desglose-completo-del-código-de-la-tarea)
9. [Respuestas Oficiales para el Informe Escrito (Rúbrica de 20 Puntos)](#9-respuestas-oficiales-para-el-informe)
10. [Simulacro de Exposición Oral: 10 Preguntas Trampa del Docente y Cómo Responder](#10-simulacro-de-exposición-oral)

---

## 1. El Gran Mapa Mental de la IA

Para no perderte en la exposición, debes tener clara la taxonomía del Machine Learning y entender por qué en la Semana 9 usamos **Regresión** mientras que en otras semanas se habla de **Árboles de Clasificación** o **K-Means**:

```
                              INTELIGENCIA ARTIFICIAL (IA)
                                           │
                                  MACHINE LEARNING (ML)
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
APRENDIZAJE SUPERVISADO                                             APRENDIZAJE NO SUPERVISADO
(Hay datos históricos con etiqueta / "target" Y)                    (No hay etiqueta Y; se buscan patrones ocultos)
         │                                                                   │
   ┌─────┴─────────────────────┐                                             ▼
   ▼                           ▼                                     CLUSTERING (Segmentación)
CLASIFICACIÓN               REGRESIÓN                                        │
(Target discreto:           (Target continuo:                                ▼
 "Sí/No", Fraude, Riesgo)    "¿Cuánto?", Salarios, Goles)                 K-MEANS
   │                           │                                    (Agrupa por distancias euclidianas
   ├─ DecisionTreeClassifier   ├─ LinearRegression (Original)        a K centroides; NO predice un valor,
   ├─ RandomForestClassifier   ├─ DecisionTreeRegressor (Alt. 1)     solo crea grupos)
   └─ LogisticRegression       ├─ RandomForestRegressor (Alt. 2)
                               └─ KNeighborsRegressor (Alt. 3)
```

### Diferencias Clave que debes dominar:
* **¿Por qué esta tarea es de Regresión y NO de Clasificación?**  
  Porque la variable a predecir es `goles_temporada`, una variable numérica continua/cuantitativa ($0, 1, 2, \dots, 35$ goles). En clasificación responderías a preguntas de categoría tipo: *"¿Anotará más de 10 goles? (Sí/No)"*. Al responder a *"¿Cuántos goles anotará?"*, es formalmente **Regresión**.
* **¿Qué diferencia hay entre esta tarea y K-Means (Semana 11)?**  
  * **En esta tarea (Supervisada):** Tenemos la respuesta correcta en cada fila (`goles_temporada`). El algoritmo compara su predicción con la realidad y ajusta sus parámetros minimizando el error cuadrático medio.
  * **En K-Means (No Supervisada):** No existe ninguna columna objetivo $y$. K-Means solo toma características $(x_1, x_2, \dots)$ y agrupa a los jugadores por similitud geométrica (por ejemplo, "delanteros rápidos", "mediocampistas pasadores"). No predice goles; segmenta perfiles.

---

## 2. Conexión con Agentes Inteligentes

¿Por qué en un curso titulado **Agentes Inteligentes** te enseñan modelos de Machine Learning como regresión y árboles?

Un **Agente Inteligente** es una entidad que percibe su entorno a través de sensores y actúa sobre él mediante actuadores para maximizar una medida de rendimiento (Russell & Norvig). El ciclo clásico es:

$$\text{Percepción (Sensores)} \longrightarrow \text{Modelo Interno / Razonamiento} \longrightarrow \text{Acción (Actuadores)}$$

1. **Percepción:** El agente recopila datos del entorno (en este caso, estadísticas de rendimiento: minutos jugados, tiros al arco, asistencias, velocidad máxima, porcentaje de pases).
2. **Modelo Interno (Machine Learning):** Un agente basado en modelos o basado en utilidad necesita predecir el impacto de sus decisiones o el rendimiento futuro. Aquí entra la **Regresión**: nuestro modelo predice cuántos goles aportará un jugador.
3. **Acción:** Con base en la predicción $\hat{y}$, el agente toma una decisión estratégica: *¿Fichamos a este delantero por \$30 millones? ¿Lo alineamos en el próximo partido titular? ¿Renovamos su contrato?*

> **Tip de Ingeniero Senior para la sustentación:**  
> Si el profesor te pregunta: *"¿Qué tiene que ver una regresión con un agente inteligente?"*, responde:  
> *"Profesor, un agente inteligente reflexivo o basado en utilidad requiere una función de predicción o modelo del mundo para estimar el estado futuro antes de ejecutar una acción. Los modelos de Machine Learning (como Random Forest o Regresión Lineal) actúan como el cerebro analítico del agente, permitiéndole inferir métricas continuas de desempeño a partir de las percepciones del entorno deportivo."*

---

## 3. Estadística Fundamental

### 3.1 Media vs Mediana: ¿Por qué importa en Datos?
En la exploración del dataset usamos `df.describe()`, donde aparecen la **media** (`mean`) y la **mediana** (el percentil `50%`).

| Concepto | Fórmula Matemática | Propiedad Fundamental | Comportamiento ante Valores Extremos (*Outliers*) |
| :--- | :--- | :--- | :--- |
| **Media Aritmética** | $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$ | Punto de equilibrio o centro de gravedad del conjunto. | **Muy sensible a valores atípicos.** Un solo Messi o Cristiano Ronaldo con 50 goles jalaría el promedio de todo el equipo hacia arriba de forma engañosa. |
| **Mediana** | Valor que divide los datos ordenados en dos mitades (50% menor, 50% mayor). | Medida de posición robusta. | **Inmune a valores atípicos.** Si agregas a un jugador de 50 goles, la mediana prácticamente no se mueve, reflejando el rendimiento del jugador típico. |

### 3.2 El Dilema Sesgo vs Varianza (*Bias-Variance Tradeoff*)
Este es el concepto teórico **MÁS IMPORTANTE** de la tarea y el motivo exacto de por qué ganó Random Forest y perdió el Árbol de Decisión individual.

```
ALTO SESGO (Underfitting / Subajuste)          EQUILIBRIO ÓPTIMO               ALTA VARIANZA (Overfitting / Sobreajuste)
        (Regresión Lineal)                     (Random Forest)                   (Decision Tree sin límite)
    
         •   •      •   •                       •   •      •   •                        •   •      •   •
           •   \   •                             •  /───\ •                             • ┌─/──\─┐ •
      ───────----\───────────────           ───────/     \──────────────           ───────/─┘    └─\─────────
            •     \   •                         • /       \   •                         •/          \ •
         •         • •                       •   •         • •                       •  •            ••
  El modelo es demasiado rígido.             Captura la tendencia real           Memoriza el ruido y los puntos
  Ignora patrones no lineales.               y generaliza a datos nuevos.        específicos del entrenamiento.
```

* **Sesgo (Bias):** Error debido a suposiciones erróneas o excesivamente simplistas en el algoritmo. Un alto sesgo hace que el modelo ignore las relaciones relevantes entre variables y la meta (**Subajuste / Underfitting**).  
  * *Ejemplo en la tarea:* La **Regresión Lineal** tiene alto sesgo para los delanteros top porque asume una línea recta fija, cuando en la realidad los goles explotan de forma no lineal con la velocidad y los tiros.
* **Varianza (Variance):** Sensibilidad del modelo a las pequeñas fluctuaciones del conjunto de entrenamiento. Un modelo con alta varianza aprende el ruido aleatorio del conjunto de datos en lugar de la señal real (**Sobreajuste / Overfitting**).  
  * *Ejemplo en la tarea:* El **Decision Tree individual sin podar** memorizó fila por fila los 400 datos de entrenamiento ($R^2 = 1.0000$, error 0), pero al probarlo con las 100 muestras nuevas de prueba, su error se disparó a $\text{RMSE} = 2.6495$.

---

## 4. Métricas de Evaluación de Regresión

En clasificación se usan métricas de matriz de confusión (*Accuracy, Precision, Recall, F1-Score*). Pero **en Regresión las predicciones son números continuos**, por lo que se miden distancias de error:

### 4.1 RMSE (Root Mean Squared Error - Error Cuadrático Medio Raíz)
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
* **¿Qué hace?** Calcula la diferencia entre el valor real $y_i$ y el valor predicho $\hat{y}_i$, la eleva al cuadrado, promedia todas las diferencias y le saca la raíz cuadrada.
* **¿Por qué la raíz cuadrada?** Porque al elevar al cuadrado las unidades se convierten en "goles al cuadrado", lo cual no tiene sentido físico. La raíz devuelve el error a la unidad original: **goles**.
* **Propiedad crítica:** **Penaliza con mayor severidad los errores grandes**. Si el modelo se equivoca por 1 gol, el cuadrado es 1. Pero si se equivoca por 6 goles, el cuadrado es 36. Es la métrica reina en analítica deportiva y finanzas cuando equivocarse por mucho es catastrófico.

### 4.2 MAE (Mean Absolute Error - Error Absoluto Medio)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
* **¿Qué hace?** Promedia directamente los valores absolutos de los errores.
* **Interpretación:** *"En promedio, el modelo se desvía en $\pm X$ goles por jugador"*. Es más robusto ante valores atípicos que el RMSE porque no eleva al cuadrado.

### 4.3 Coeficiente de Determinación ($R^2$)
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
* **¿Qué significa?** Representa la **proporción de la varianza total** de la variable objetivo ($y$) que es explicada por las variables predictoras ($X$) del modelo.
* **Escala:**
  * $R^2 = 1.0$ (100%): Predicción matemáticamente perfecta.
  * $R^2 = 0.0$: El modelo no predice nada mejor que simplemente predecir siempre el promedio $\bar{y}$.
  * $R^2 < 0.0$: El modelo es tan pésimo que predecir la media histórica daría mejores resultados.
* **En nuestra tarea:**
  * Random Forest alcanzó $R^2 = 0.9388$ en Test $\rightarrow$ Explica el **93.88%** del comportamiento goleador en jugadores que jamás vio durante el entrenamiento.

---

## 5. Pandas y Preprocesamiento de Datos

En el script `build_sem09.py` y en el notebook ejecutamos el flujo estándar de la industria:

### 5.1 Carga e Inspección con Pandas
```python
import pandas as pd

df = pd.read_csv("football_players_goals_kaggle.csv")
print("Dimensiones:", df.shape)  # Retorna una tupla: (500, 6) -> 500 filas (jugadores), 6 columnas
display(df.head())              # Muestra las primeras 5 filas para inspeccionar tipos y valores
display(df.describe().round(2)) # Muestra conteo, media, std, min, 25%, 50% (mediana), 75%, max
```

### 5.2 Separación en Matriz $X$ y Vector $y$
```python
# Variables predictoras (features): todas las columnas excepto el target
FEATURES = [c for c in df.columns if c != "goles_temporada"]
X = df[FEATURES]              # Matriz de 500 filas x 5 columnas
y = df["goles_temporada"]     # Vector Serie unidimensional de 500 valores
```
* **$X$ (mayúscula):** Por convención algebraica, es una matriz bidimensional (muestras $\times$ características).
* **$y$ (minúscula):** Es un vector unidimensional con los valores objetivo reales.

### 5.3 Partición Train / Test: 80% Entrenamiento y 20% Prueba
```python
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
```
* **¿Por qué dividimos los datos?** Si evalúas a un alumno con el mismo examen exacto que le diste para estudiar, no sabes si aprendió a razonar o si simplemente memorizó las respuestas. Dividir en `Train` y `Test` simula el mundo real: evaluar el modelo en **datos nunca antes vistos**.
* **¿Qué significa `test_size=0.2`?** El 20% (100 jugadores) se reservan para el examen final (`Test`), y el 80% (400 jugadores) se usan para entrenar (`Train`).
* **¿Qué hace `random_state=42`?** Fija la semilla del generador pseudoaleatorio. Garantiza **reproducibilidad científica**: cualquier persona en cualquier máquina obtendrá exactamente las mismas 400 filas de entrenamiento y las mismas 100 de prueba.

### 5.4 Estandarización con `StandardScaler` (Prevención de Fuga de Datos)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr) # Calcula media y desviación en TRAIN y transforma
X_te_s = scaler.transform(X_te)     # SOLO transforma TEST usando la media y desviación de TRAIN
```
* **Fórmula del Z-score:**
  $$z = \frac{x - \mu}{\sigma}$$
  Donde $\mu$ es la media y $\sigma$ la desviación estándar. Convierte todas las variables para que tengan **media = 0** y **varianza = 1**.
* **¿Por qué `fit_transform` en Train pero solo `transform` en Test?**  
  ¡Pregunta fija de examen! Si ejecutas `fit` sobre `X_te`, estás calculando la media y varianza del conjunto de prueba, lo que constituye **Data Leakage (Fuga de Información)**: tu modelo estaría haciendo "trampa" al enterarse de la distribución de los datos futuros antes de predecir.

---

## 6. Los 4 Algoritmos Evaluados

### 1. Linear Regression (Modelo Original del Autor)
* **Principio:** Ajusta una ecuación lineal de hiperplano:
  $$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k$$
  Calcula los coeficientes $\beta$ mediante el método de Mínimos Cuadrados Ordinarios (OLS), minimizando la suma de errores al cuadrado.
* **Comportamiento en la tarea:** Tuvo un desempeño respetable ($R^2 = 0.7670$, $\text{RMSE} = 3.2876$), pero falló en el tercio superior de jugadores goleadores. No puede doblar su superficie para captar efectos multiplicativos.

### 2. Decision Tree Regressor (Alternativo 1)
* **Principio:** Divide recursivamente el espacio de datos en hiperrectángulos ortogonales mediante preguntas binarias (ej: `¿tiros_al_arco > 2.6?`). En cada región final (hoja), la predicción es el **promedio de los valores $y$** de las muestras que cayeron en esa hoja.
* **Criterio de corte:** Minimiza la varianza interna o el error cuadrático medio dentro de cada nodo hijo.
* **Comportamiento en la tarea:** **Overfitting de libro de texto**. Al no fijarle un límite de profundidad (`max_depth=None`), el árbol creó tantas ramas que cada jugador de entrenamiento terminó en una hoja propia. Resultado: $R^2 = 1.0000$ y $\text{RMSE} = 0.0$ en Train, pero al ver datos nuevos en Test su error fue de $\text{RMSE} = 2.6495$.

### 3. Random Forest Regressor (Alternativo 2 - El Campeón)
* **Principio:** Es un **ensamble de árboles** basado en **Bagging (Bootstrap Aggregating)** y selección aleatoria de variables (*Feature Bagging*):
  1. Genera 100 subconjuntos de datos mediante muestreo aleatorio con reemplazo (*Bootstrap*).
  2. Entrena 100 árboles de decisión independientes. En cada corte de cada árbol, solo evalúa un subconjunto aleatorio de variables.
  3. Para predecir, hace pasar al jugador por los 100 árboles y **promedia sus 100 respuestas**:
     $$\hat{y}_{\text{bosque}} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$
* **¿Por qué triunfa?** La ley de los grandes números y la agregación demuestran que promediar modelos no correlacionados reduce drásticamente la varianza del error sin aumentar el sesgo. Obtuvo un impresionante **$R^2 = 0.9388$** y un bajísimo **$\text{RMSE} = 1.6852$ goles**.

### 4. K-Nearest Neighbors Regressor (Alternativo 3 - Extra)
* **Principio:** Es un algoritmo basado en instancias (*Lazy Learning*). No "entrena" una fórmula matemática. Guarda los datos de entrenamiento en memoria.
* Cuando llega un nuevo jugador de prueba:
  1. Calcula la distancia euclidiana estandarizada hacia todos los jugadores del conjunto de entrenamiento.
  2. Identifica a los $K=5$ jugadores más parecidos.
  3. La predicción es el **promedio aritmético** de los goles anotados por esos 5 vecinos.
* **Comportamiento en la tarea:** Desempeño sólido ($R^2 = 0.8575$, $\text{RMSE} = 2.5712$), superando a la Regresión Lineal gracias a su capacidad de captar agrupaciones locales.

---

## 7. Análisis de los Resultados Experimentales

Estos son los valores exactos generados por el código y registrados en el notebook:

### Tabla Comparativa Consolidada
| Modelo | RMSE en Train | RMSE en Test (Métrica Principal ↓) | MAE en Test (↓) | $R^2$ en Train | $R^2$ en Test (↑) | Diagnóstico Técnico |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression (Original)** | 3.3438 | **3.2876** | 2.5411 | 0.8135 | **0.7670 (76.70%)** | Subajuste moderado (alto sesgo) ante patrones no lineales |
| **Decision Tree Regressor (Alt. 1)** | 0.0000 | **2.6495** | 1.8800 | 1.0000 | **0.8487 (84.87%)** | **Sobreajuste severo (Overfitting)**: memorizó el entrenamiento |
| **Random Forest Regressor (Alt. 2)** | 0.6296 | **1.6852** | 1.3030 | 0.9934 | **0.9388 (93.88%)** | **Ganador indiscutible**: mínima varianza y excelente generalización |
| **KNN Regressor ($K=5$) (Alt. 3)** | 2.1875 | **2.5712** | 1.6440 | 0.9202 | **0.8575 (85.75%)** | Buen desempeño local, pero limitado en los extremos |

---

## 8. Desglose Completo del Código de la Tarea

Revisemos las secciones del código para que sepas qué responder si te piden señalar una línea:

### Generación o Carga de Datos y Visualización
```python
# Carga el archivo CSV con pandas
df = pd.read_csv(ARCHIVO_CSV)
print("Dimensiones del dataset:", df.shape) # Muestra (500, 6)
```

### El Loop de Entrenamiento y Evaluación Automatizada
```python
modelos = {
    "Modelo Original: Linear Regression": LinearRegression(),
    "Alternativo 1: Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Alternativo 2: Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Alternativo 3: KNN Regressor (K=5)": KNeighborsRegressor(n_neighbors=5)
}

resultados = []
predicciones_test = {}

# Itera automáticamente por cada par clave-valor del diccionario
for nombre, modelo in modelos.items():
    modelo.fit(X_tr_s, y_tr)                  # 1. Ajusta los parámetros del modelo con datos de TRAIN estandarizados
    pred_tr = modelo.predict(X_tr_s)          # 2. Predice sobre TRAIN para verificar sobreajuste
    pred_te = modelo.predict(X_te_s)          # 3. Predice sobre TEST (datos que el modelo NUNCA ha visto)
    predicciones_test[nombre] = pred_te
    
    # 4. Almacena las métricas calculadas con scikit-learn
    resultados.append({
        "Modelo": nombre,
        "RMSE (Train)": np.sqrt(mean_squared_error(y_tr, pred_tr)),
        "RMSE (Test) ↓": np.sqrt(mean_squared_error(y_te, pred_te)),
        "MAE (Test) ↓": mean_absolute_error(y_te, pred_te),
        "R² (Train)": r2_score(y_tr, pred_tr),
        "R² (Test) ↑": r2_score(y_te, pred_te)
    })
```
* **`np.sqrt(mean_squared_error(...))`:** La función `mean_squared_error` devuelve el MSE. Al envolverla en `np.sqrt()`, obtenemos el RMSE en la escala real de goles.
* **`r2_score(y_te, pred_te)`:** Compara la varianza residual con la varianza total de los goles reales de prueba.

---

## 9. Respuestas Oficiales para el Informe

Usa estas redacciones exactas para tu informe en PDF y para tu presentación:

### Pregunta 1: ¿Qué modelo obtuvo el mejor resultado?
> **Respuesta:**  
> El **Modelo Alternativo 2 (`Random Forest Regressor`)** obtuvo el mejor resultado indiscutible en el conjunto de prueba (`Test`), logrando el menor error cuadrático medio raíz (**$\text{RMSE} = 1.6852$ goles**), el menor error absoluto medio (**$\text{MAE} = 1.3030$ goles**) y el mayor coeficiente de determinación (**$R^2 = 0.9388$**, explicando el **93.88%** de la varianza total). Superó con contundencia al modelo de Regresión Lineal del autor original ($R^2 = 0.7670$), demostrando una extraordinaria capacidad de generalización sobre jugadores nunca antes vistos.

### Pregunta 2: ¿Qué modelo obtuvo el peor resultado?
> **Respuesta:**  
> Evaluando la capacidad de predicción en datos reales de prueba, el **Modelo Original (`Linear Regression`)** tuvo el peor error global (**$\text{RMSE} = 3.2876$ goles** y $R^2 = 0.7670$), debido a su incapacidad matemática para amoldarse a relaciones no lineales.  
> Sin embargo, a nivel metodológico, el **Modelo Alternativo 1 (`Decision Tree Regressor`)** presentó el peor comportamiento cualitativo al sufrir de un **sobreajuste extremo (Overfitting)**: en el conjunto de entrenamiento alcanzó un $R^2 = 1.0000$ perfecto con $\text{RMSE} = 0.0$ (memorizó cada jugador), pero al enfrentarse a las muestras de prueba su desempeño cayó estrepitosamente a $\text{RMSE} = 2.6495$ y $R^2 = 0.8487$, evidenciando una alta varianza e inestabilidad.

### Pregunta 3: ¿Las diferencias fueron grandes o pequeñas?
> **Respuesta:**  
> Las diferencias fueron **grandes, estadísticamente significativas y de alto impacto práctico en la toma de decisiones**.  
> Comparando el modelo original con el mejor modelo:
> * El error RMSE se redujo de **$3.2876$ a $1.6852$ goles**, lo que representa una **disminución del $48.7\%$ del error original** (el error se cortó casi a la mitad).
> * La capacidad predictiva ($R^2$) aumentó en **$+17.18$ puntos porcentuales** (de $76.70\%$ a $93.88\%$).  
> En el fútbol profesional, errar por 3.3 goles por temporada puede significar fichar erróneamente a un delantero suplente como si fuera estrella; reducir ese margen a 1.6 goles proporciona una precisión analítica confiable para la gerencia deportiva.

### Pregunta 4: ¿Por qué cree que algunos modelos funcionaron mejor que otros para este problema?
> **Respuesta:**  
> 1. **Naturaleza no lineal y sinérgica del fútbol:** El éxito goleador no depende de la suma aislada de atributos, sino de interacciones multiplicativas complejas. Por ejemplo, tener alta velocidad (`velocidad_max_kmh > 30.5`) combinada con alto volumen de remates (`tiros_al_arco > 2.1`) produce una explosión goleadora que una recta no puede modelar.
> 2. **Falla de la Regresión Lineal (Alto Sesgo):** Al imponer una ecuación lineal rígida ($\hat{y} = \beta_0 + \sum \beta_i x_i$), asume que cada tiro al arco aporta siempre el mismo incremento constante sin importar el contexto físico del jugador, subestimando a los goleadores de élite.
> 3. **Falla del Árbol de Decisión (Alta Varianza):** Puede modelar no linealidades, pero al crecer libremente crea reglas hiperespecíficas para una sola muestra, memorizando el ruido estocástico del entrenamiento.
> 4. **Éxito del Bosque Aleatorio (Random Forest):** Al combinar 100 árboles con subconjuntos aleatorios de datos (*bagging*) y de variables (*feature bagging*), conserva la flexibilidad no lineal de los árboles pero cancela sus errores individuales al promediarlos, encontrando el punto óptimo del dilema sesgo-varianza.

---

## 10. Simulacro de Exposición Oral

Prepárate con estas 10 preguntas que los profesores de la USIL suelen hacer en la sustentación:

#### P1: "¿Por qué utilizaste `random_state=42` tanto en el split como en los modelos?"
* **Respuesta Senior:** *"El parámetro `random_state` actúa como la semilla del generador pseudoaleatorio. Se utiliza para garantizar la reproducibilidad experimental: permite que cualquier evaluador que ejecute el código obtenga exactamente la misma partición 80/20 y las mismas combinaciones de árboles en Random Forest, cumpliendo con el rigor del método científico."*

#### P2: "¿Por qué en el Decision Tree el $R^2$ en Train dio 1.0000 y el RMSE dio 0?"
* **Respuesta Senior:** *"Porque no le configuramos un hiperparámetro de poda como `max_depth` o `min_samples_split`. El árbol creció de forma codiciosa (*greedy*) hasta que cada hoja contuvo una sola muestra de entrenamiento. Al no haber impureza en las hojas, la predicción es idéntica al valor real, memorizando los datos. Es el síntoma clásico de sobreajuste o sobreentrenamiento."*

#### P3: "¿Por qué estandarizaste con `StandardScaler` si los Árboles y Random Forest no lo necesitan?"
* **Respuesta Senior:** *"Es correcto que los árboles son invariantes a transformaciones monotónicas porque evalúan umbrales de corte por variable individual. Sin embargo, mantuvimos el mismo preprocesamiento uniforme porque el modelo original era una Regresión Lineal y uno de nuestros alternativos era KNN. KNN calcula distancias euclidianas, por lo que si una variable como minutos (hasta 3400) no se normaliza, aplastaría a los tiros al arco (hasta 4.5). Mantener el mismo preprocesamiento garantiza una comparación metodológicamente justa y rigurosa bajo las mismas condiciones."*

#### P4: "¿Qué es el Data Leakage y cómo lo evitaste en el código?"
* **Respuesta Senior:** *"La fuga de datos (Data Leakage) ocurre cuando información del conjunto de prueba se filtra involuntariamente en el entrenamiento. Lo evitamos al aplicar `scaler.fit_transform(X_tr)` únicamente sobre el 80% de entrenamiento (calculando la media y desviación solo de Train) y luego ejecutando `scaler.transform(X_te)` sobre Test sin recalcular estadísticas."*

#### P5: "¿Cuál es la diferencia entre Bagging y Boosting?"
* **Respuesta Senior:** *"Random Forest utiliza Bagging (Bootstrap Aggregating): entrena múltiples árboles en paralelo de manera independiente y promedia sus salidas para reducir la varianza. En cambio, Boosting (como XGBoost o Gradient Boosting) entrena árboles en serie de manera secuencial, donde cada nuevo árbol intenta corregir los errores residuales del árbol anterior para reducir principalmente el sesgo."*

#### P6: "¿Por qué no usaron Accuracy o F1-Score como métricas?"
* **Respuesta Senior:** *"Porque Accuracy, Precision, Recall y F1-Score son métricas exclusivas para problemas de Clasificación (donde evaluamos aciertos categóricos en una matriz de confusión). En Regresión, la variable dependiente es continua y no tiene sentido hablar de 'acierto binario exacto', sino de magnitud de distancia del error, evaluada formalmente con RMSE, MAE y $R^2$."*

#### P7: "¿Qué pasaría si cambiamos el valor de $K$ en KNN de 5 a 1?"
* **Respuesta Senior:** *"Si $K=1$, el modelo predice asignando exactamente el valor del vecino más cercano. En el conjunto de entrenamiento el error sería 0 ($R^2=1.0$), pero sufriría de sobreajuste severo y altísima varianza, siendo hipersensible al ruido. Si $K$ fuera muy grande (ej. $K=400$), caería en alto sesgo, prediciendo simplemente el promedio de todo el dataset."*

#### P8: "¿Qué ventaja tiene el MAE sobre el RMSE al hablar con un director técnico de fútbol?"
* **Respuesta Senior:** *"El MAE tiene una interpretación intuitiva y lineal: un MAE de 1.30 goles significa que en promedio nos equivocamos por aproximadamente 1 gol por jugador en la temporada. El RMSE, en cambio, penaliza cuadráticamente los errores graves, por lo que es más técnico pero alerta mejor si el modelo tuvo pifias gigantescas en algún caso particular."*

#### P9: "¿Por qué Random Forest es considerado un modelo tipo 'caja negra' en comparación con la Regresión Lineal?"
* **Respuesta Senior:** *"En la Regresión Lineal tenemos una fórmula analítica directa donde cada coeficiente $\beta_i$ indica el cambio exacto en goles por cada unidad adicional de la variable. En Random Forest, la predicción proviene del promedio de 100 árboles con cientos de ramificaciones no lineales, lo que dificulta extraer una ecuación simple, requiriendo técnicas como Feature Importance o valores SHAP para su interpretación."*

#### P10: "¿Cómo implementarías este modelo en un agente inteligente autónomo?"
* **Respuesta Senior:** *"El agente dispondría de un módulo sensor que ingesta datos de la API de rendimiento deportivo tras cada jornada. Dicho vector pasa por el pipeline de estandarización y por el modelo entrenado de Random Forest. La inferencia resultante alimenta la función de utilidad del agente, permitiéndole enviar alertas automáticas al cuerpo técnico para recomendar rotaciones o sugerir contrataciones en el mercado de pases basándose en goles esperados."*

