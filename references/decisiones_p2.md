# Decisiones de P2: riesgo de vaciado o llenado de estaciones

Decisiones específicas de P2: ¿una estación estará en riesgo de quedarse sin bicicletas o sin
anclajes disponibles en un periodo futuro? Las decisiones generales del proyecto (formato de
los datos, periodo de análisis) están en [decisiones.md](decisiones.md).

## 1. Clasificación de 3 clases: sin riesgo, se vacía, se llena

- **Fecha:** 2026-10-16
- **Estado:** vigente
- **Aplica a:** P2 (riesgo de quedarse sin bicicletas o sin anclajes).

### Contexto

P2 pregunta si una estación estará en riesgo de quedarse sin bicicletas o sin anclajes
disponibles. El ML Canvas (iteración 1) la planteaba como clasificación binaria: riesgo sí/no.

### Decisión

Clasificar cada estación y periodo en **3 clases**:

| Clase | Significado | Acción del equipo de rebalanceo |
|---|---|---|
| Sin riesgo | La estación opera normal | Ninguna |
| Se vacía | Salen más bicicletas de las que llegan; puede quedarse sin bicicletas | Llevar bicicletas |
| Se llena | Llegan más bicicletas de las que salen; puede quedarse sin anclajes | Retirar bicicletas |

### Por qué

- **Las acciones son opuestas.** Con la clasificación binaria el equipo sabría a qué estación
  ir, pero no si debe llegar con bicicletas o con espacio para recogerlas.
- **Los datos lo permiten.** El riesgo se aproxima con el balance neto de viajes
  (llegadas − salidas), cuyo signo indica si la estación se vacía o se llena.
- **Los modelos candidatos no cambian.** `LogisticRegression` y
  `HistGradientBoostingClassifier` de scikit-learn manejan 3 clases igual que 2 y dan
  probabilidades por clase (`predict_proba`) para ordenar la lista de estaciones.

### Alternativa descartada

**Clasificación binaria (riesgo / sin riesgo).** Es más simple, pero pierde la dirección del
riesgo, que es justo lo que el equipo de rebalanceo necesita para actuar.

### Consecuencias

- Las clases de riesgo serán minoritarias (la mayor parte del tiempo las estaciones operan
  normal). La exactitud (accuracy) no sirve como métrica: se usan recall y PR-AUC por clase, y
  pesos de clase (`class_weight`) en los modelos.
- El baseline de clase mayoritaria tendrá recall de 0 en las clases de riesgo; sirve como
  piso, no como referencia útil.
- Hay que actualizar la sección de P2 del ML Canvas (`references/MLCanvas_MiBici.pdf`).
- **Pendiente:** definir con qué umbral el balance neto frente a la capacidad cuenta como
  "se vacía" o "se llena".

## 2. Predecir el riesgo en ventanas de 2 horas

- **Fecha:** 2026-10-17
- **Estado:** vigente
- **Aplica a:** la etiqueta, las variables y las predicciones de P2.

### Contexto

Las instrucciones de la actividad dejan la granularidad al equipo ("en un periodo futuro"). El
ML Canvas (iteración 1) proponía ventanas de 2 horas sin justificarlo y, en otra sección,
hablaba del riesgo "en el próximo día". El análisis está en
`notebooks/2.0-rc-p2-tamano-ventana.ipynb`.

### Decisión

Predecir, para cada estación, la clase de riesgo en **ventanas de 2 horas** que empiezan en
hora par (00:00, 02:00, …, 22:00).

### Por qué

Se compararon ventanas de 30 minutos, 1, 2, 3 y 4 horas y el día completo; todas las cifras
de esta sección salen del notebook. Para comparar con la
misma regla se usó un umbral provisional: riesgo cuando el balance neto
(llegadas − salidas) de la ventana pasa del 50% de la capacidad.

| Medida | Resultado | Qué implica |
|---|---|---|
| Horas pico | En días hábiles, de 7 a 9 h y de 17 a 19 h; duran ~2–3 h | Ventanas de 2 o 3 h siguen la forma de los picos |
| Rareza del riesgo | 30 min: 0.04% de las ventanas; 1 h: 0.2%; 2 h: ~0.9% (~60 mil ejemplos por clase); 3 h: ~1.8%; día: ~11% | Descarta 30 min y 1 h: muy pocos ejemplos de riesgo |
| Ambos riesgos el mismo día | Con ventanas de 2 a 4 h, 16–19% de los días con riesgo tienen los dos tipos (10% con 30 min, 11% con 1 h) | Descarta el día completo: no puede representar "se vacía en la mañana y se llena en la tarde" ni decir a qué hora actuar |
| Bloques en hora par o impar | Sin diferencia (0.86% contra 0.89%) | No importa dónde empiecen; se usan los pares, que es lo que hace `dt.floor` |
| Duración de los viajes | Mediana de 9.4 min. Terminan en otra ventana: 35% con 30 min, 17.5% con 1 h, 8.6% con 2 h | Con 2 h, más del 91% de los viajes empieza y termina en la misma ventana |
| Duración de los eventos de desbalance | En ventanas con riesgo, el mejor sub-bloque concentra (mediana) el 75% del desbalance si es de 1 h, 85–88% si es de 2 h y 100% si es de 3 h | Un evento dura entre 2 y 3 horas: 1 h lo corta; 2 h captura su núcleo; 3 h lo captura completo |

Los datos descartan 30 minutos, 1 hora y el día completo, y muestran que un evento de
desbalance dura entre 2 y 3 horas. Entre esas dos opciones, **se eligen 2 horas porque el
objetivo de P2 es saber cuándo una estación estará en riesgo**: 2 horas captura el núcleo del
evento (85–88% del desbalance) con más precisión sobre el momento que 3 horas, que captura el
evento completo pero diluye cuándo ocurre. Además deja ~60 mil ejemplos por clase de riesgo y
coincide con el ML Canvas.

### Alternativas descartadas

- **30 minutos o 1 hora:** riesgo demasiado raro y muchos viajes que terminan en otra ventana. Además, predecir con un día de anticipación a
  qué media hora ocurrirá el problema pide una precisión que el modelo no puede dar.
- **3 horas:** captura el evento completo y da más ejemplos de riesgo, pero con menos precisión
  sobre cuándo ocurre, que es lo que el equipo de rebalanceo necesita.
- **4 horas:** lo mismo que 3 horas, con todavía menos precisión.
- **Día completo:** no dice a qué hora actuar y pierde los días en que una estación se vacía y
  se llena.

### Consecuencias

- 12 ventanas por estación por día.
- La ventana de predicción (2 h) no obliga a calcular la etiqueta con esa resolución: el
  inventario se puede simular viaje por viaje y marcar la ventana como "se vacía" o "se llena"
  si en cualquier momento dentro de ella se cruza el umbral.
- Los porcentajes de arriba usan el umbral provisional; con la etiqueta final cambiarán.
- **Pendiente para la decisión de la etiqueta:** con cuántas bicicletas se supone que empieza
  cada estación y qué umbral cuenta como riesgo.
- **Pendiente de revisión:** 3.6% de los viajes dura menos de 1 minuto y 45 duran más de
  24 horas.

## 3. Baselines y modelos candidatos

- **Fecha:** 2026-10-17
- **Estado:** vigente
- **Aplica a:** el entrenamiento y la evaluación de P2.

### Contexto

La actividad pide, para cada pregunta, un baseline sencillo y al menos un modelo candidato. En
esta entrega solo los baselines se implementan y se registran en MLflow; los candidatos se
proponen y justifican. El ML Canvas (iteración 1) proponía como baseline la clase mayoritaria y
como candidatos regresión logística y gradient boosting con pesos de clase.

### Decisión

| Rol | Modelo | Qué predice |
|---|---|---|
| Baseline 1 | Clase mayoritaria (`DummyClassifier(strategy="most_frequent")`) | Siempre "sin riesgo" |
| Baseline 2 | Persistencia semanal | La misma clase que tuvo la estación en la misma ventana y el mismo día de la semana anterior |
| Candidato 1 | `LogisticRegression(class_weight="balanced")` | Probabilidad de cada clase a partir de las variables |
| Candidato 2 | `HistGradientBoostingClassifier(class_weight="balanced")` | Probabilidad de cada clase a partir de las variables |

Los cuatro se evalúan con las mismas métricas sobre el conjunto de prueba: recall y PR-AUC por
clase de riesgo ("se vacía" y "se llena"), no exactitud (ver decisión 1).

### Por qué

- **Clase mayoritaria:** es el piso mínimo. Como las clases de riesgo son ~1% de las ventanas,
  tiene exactitud alta pero recall de 0 en las clases de riesgo: muestra por qué la exactitud no
  sirve como métrica.
- **Persistencia semanal:** es una regla simple que sí detecta riesgo, así que es la referencia
  real que un modelo debe superar. Se usa la semana anterior y no el día anterior porque los
  días hábiles y los fines de semana tienen patrones distintos, y porque es el mismo tipo de
  baseline que el ML Canvas propone para P1. Solo usa información disponible al momento de
  predecir (la noche anterior).
- **`LogisticRegression`:** simple, rápida e interpretable (los coeficientes indican qué variable
  pesa más). Sirve para saber si una relación aproximadamente lineal entre las variables y el
  riesgo es suficiente.
- **`HistGradientBoostingClassifier`:** captura relaciones no lineales y combinaciones de
  variables (por ejemplo, hora pico en día hábil en una estación del centro) y está diseñado para
  conjuntos grandes, como los ~20 millones de viajes agregados por estación y ventana.
- **Ambos candidatos** manejan 3 clases, dan probabilidades por clase (`predict_proba`) para
  ordenar la lista diaria de estaciones y aceptan `class_weight` para compensar que las clases de
  riesgo sean minoritarias.

### Alternativas descartadas

- **Solo la clase mayoritaria como baseline:** con recall de 0 no sirve como referencia; cualquier
  modelo que detecte algo de riesgo la supera.
- **SVM y k vecinos más cercanos:** muy lentos con millones de filas, y SVM no da probabilidades
  directamente.

### Consecuencias

- Para entrenar los modelos hace falta primero la etiqueta (inventario inicial y umbral) y las
  variables de `features.py`.
- La persistencia semanal no puede predecir la primera semana de datos; esas ventanas se excluyen
  de la evaluación o caen antes del periodo de prueba.
