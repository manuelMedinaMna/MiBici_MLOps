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
- El umbral que define "se vacía" y "se llena" está en la decisión 5.

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
- El inventario inicial y el umbral de la etiqueta final están en las decisiones 4 y 5.
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
| Baseline 2 | Frecuencia en las últimas 4 semanas | Para cada clase de riesgo, la fracción de las últimas 4 semanas (misma estación, mismo día de la semana y misma ventana) en que hubo ese riesgo: 0, 0.25, 0.5, 0.75 o 1 |
| Candidato 1 | `LogisticRegression(class_weight="balanced")` | Probabilidad de cada clase a partir de las variables |
| Candidato 2 | `HistGradientBoostingClassifier(class_weight="balanced")` | Probabilidad de cada clase a partir de las variables |

Los cuatro se evalúan con las mismas métricas sobre el conjunto de prueba: recall y PR-AUC por
clase de riesgo ("se vacía" y "se llena"), no exactitud (ver decisión 1).

### Por qué

- **Clase mayoritaria:** es el piso mínimo. Como las clases de riesgo son ~1% de las ventanas,
  tiene exactitud alta pero recall de 0 en las clases de riesgo: muestra por qué la exactitud no
  sirve como métrica.
- **Frecuencia en las últimas 4 semanas:** es una regla simple que sí detecta riesgo, así que es
  la referencia real que un modelo debe superar. Solo usa información disponible al momento de
  predecir (la noche anterior).
  - **Mismo día de la semana, no el día anterior:** los días hábiles y los fines de semana tienen
    patrones distintos; un lunes se parece más al lunes pasado que al domingo.
  - **4 semanas, no solo la anterior:** una semana atípica (feriado, lluvia) pesa solo 1/4.
    Además da un puntaje (0 a 1) y no solo "sí/no", así que se puede calcular la PR-AUC igual
    que con los modelos; con una sola semana la PR-AUC casi no tendría sentido.
  - **4 y no todo el historial:** reacciona a cambios recientes, como las estaciones nuevas de
    2026. Cuatro semanas son aproximadamente un mes; al ser un baseline no se optimiza.
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
- El baseline de frecuencia no puede predecir las primeras 4 semanas de datos; esas ventanas
  quedan antes del periodo de prueba.
- Las mismas variables (riesgo hace 1 día, hace 1 semana, frecuencia en las últimas 4 semanas)
  pueden servir como variables de entrada de los modelos candidatos en `features.py`.

## 4. Inventario inicial: reinicio diario al 50% de la capacidad

- **Fecha:** 2026-10-17
- **Estado:** vigente
- **Aplica a:** el cálculo de la etiqueta de P2.

### Contexto

MiBici no publica cuántas bicicletas hay en cada estación, así que el inventario se simula: se
supone un inventario inicial y se le suman las llegadas y se le restan las salidas. Tampoco se
sabe cuándo pasa el camión de rebalanceo, así que la simulación tiene que reiniciarse cada cierto
tiempo; si se acumulara sin reinicio, el inventario estimado crecería o bajaría sin límite.

### Decisión

Cada día **a las 4:00**, cada estación **empieza con el 50% de su capacidad** (`dpcapacity`).
El "día" de la simulación va de las 4:00 a las 3:59 del día siguiente. Durante el resto del día el inventario estimado se **acumula**: al inventario
inicial se le suman las llegadas y se le restan las salidas, y no se reinicia en cada ventana de
2 horas.

**Ejemplo:** estación con capacidad 20, empieza con 10 bicicletas.

| Ventana | Llegadas − salidas | Inventario estimado al final |
|---|---|---|
| 6–8 h | −6 | 10 − 6 = 4 |
| 8–10 h | −6 | 4 − 6 = −2 → se vacía |

### Por qué

- **Reinicio diario:** supone que en la noche, cuando casi no hay viajes, el sistema se rebalancea,
  que es la práctica usual. Es la forma más simple de evitar que el inventario estimado se
  descontrole sin conocer los recorridos reales de rebalanceo.
- **A las 4:00:** entre la 1:00 y las 5:00 casi no hay viajes (0 por día de 2:00 a 4:59), así
  que el sistema prácticamente no opera en ese horario. A las 00:00 hay ~23 viajes por día a medio
  camino que el reinicio cortaría; a las 4:00, prácticamente ninguno (0.4 por día). Es la última
  hora sin actividad antes de que arranque el servicio y coincide con el inicio de un bloque de
  2 horas (04:00–06:00). Cualquier hora entre 2:00 y 4:00 daría el mismo resultado. Las cifras
  están en la sección 11 de `notebooks/2.0-rc-p2-tamano-ventana.ipynb`.
- **50% de la capacidad:** no favorece ni el vaciado ni el llenado; las dos clases de riesgo
  parten de la misma distancia.
- **Acumular dentro del día:** el riesgo de una ventana depende de lo que pasó antes ese mismo
  día. En el ejemplo, la estación se vacía a las 8–10 h porque ya venía perdiendo bicicletas desde
  las 6; si cada ventana empezara de nuevo al 50%, ese riesgo no se detectaría.

### Alternativas descartadas

- **Reiniciar al 50% en cada ventana de 2 horas:** cada ventana "olvida" lo que pasó antes. Es lo
  que se usó como umbral provisional en `notebooks/2.0-rc-p2-tamano-ventana.ipynb`, suficiente
  para comparar tamaños de ventana con la misma regla, pero no para la etiqueta final.
- **Sin reinicio:** sin observar el rebalanceo, el inventario estimado llega a cientos de
  bicicletas negativas o positivas.
- **Inventario inicial distinto por estación, estimado de los datos:** más realista (una estación
  que se vacía cada mañana probablemente empieza llena), pero más compleja y difícil de validar
  sin el inventario real. Queda como posible mejora.

### Consecuencias

- Los porcentajes de riesgo de la decisión 2 se calcularon reiniciando cada ventana; con esta
  regla cambiarán. La elección de la ventana de 2 horas no depende de eso, porque todos los
  tamaños se compararon con la misma regla.
- El umbral que define "se vacía" y "se llena" está en la decisión 5.

## 5. Umbral de riesgo: llegar a 0 bicicletas o a la capacidad

- **Fecha:** 2026-10-17
- **Estado:** vigente
- **Aplica a:** el cálculo de la etiqueta de P2.

### Contexto

Con el inventario simulado de la decisión 4, falta decidir a partir de qué nivel una ventana
cuenta como riesgo. Se compararon umbrales de 0%, 10%, 20% y 30% de la capacidad, simulando el
inventario de todas las estaciones desde 2022-03 y contando también las ventanas sin viajes (si
una estación quedó vacía y nadie llegó después, sigue vacía).

### Decisión

En cada ventana de 2 horas se toma el inventario simulado más bajo y el más alto que alcanzó la
estación:

| Clase | Regla |
|---|---|
| Se vacía | El inventario más bajo llega a 0 o menos |
| Se llena | El inventario más alto llega a la capacidad o más |
| Sin riesgo | Ninguna de las dos |

La etiqueta se interpreta como **riesgo sin intervención**: "si la estación empezara el día a la
mitad y nadie la rebalanceara, en esta ventana se quedaría sin bicicletas (o sin anclajes)".

### Por qué

| Umbral | Se vacía | Se llena | Total en riesgo |
|---|---|---|---|
| **0%** | 9.8% | 8.8% | **~19%** |
| 10% | 15.2% | 12.6% | ~28% |
| 20% | 21.2% | 16.7% | ~38% |
| 30% | 32.9% | 24.1% | ~57% |

- **Con 20% o 30%, entre el 38% y el 57% de las ventanas serían alerta.** Una alerta que aparece
  casi la mitad del tiempo no ayuda al equipo de rebalanceo a decidir a dónde ir primero.
- **Con 0% la etiqueta tiene un significado claro** ("sin intervención, se queda vacía o llena")
  y las clases quedan minoritarias pero no rarísimas, repartidas casi igual entre "se vacía" y
  "se llena".
- **La anticipación no depende del umbral.** La predicción se hace la noche anterior para el día
  siguiente, y el equipo actúa según la probabilidad que da el modelo, no según el hecho de que la
  estación ya esté vacía.

### Hallazgo: el inventario simulado sale de lo posible

En el 9.8% de las ventanas el inventario simulado queda por debajo de 0 y en el 8.8% por encima
de la capacidad, lo que en la realidad no puede pasar (nadie saca una bicicleta de una estación
vacía). Indica que durante el día alguien rebalanceó la estación o que empezó con más o menos
del 50%. Por eso la etiqueta no dice que la estación se quedó vacía, sino que se quedaría vacía
sin intervención, que es justo lo que el equipo de rebalanceo necesita saber.

### Alternativas descartadas

- **Umbrales de 10% a 30%:** más anticipación en la etiqueta, pero demasiadas ventanas en alerta.
- **Dos niveles de alerta en la etiqueta (alerta y alerta roja):** convertiría las 3 clases en 5 y
  haría más raras las clases de riesgo. La severidad se obtiene de la probabilidad del modelo:

  | Probabilidad de la clase de riesgo | Alerta |
  |---|---|
  | ≥ 70% | Roja: ir primero |
  | 40–70% | Amarilla: ir si hay tiempo |
  | < 40% | Sin acción |

  Estos cortes son una propuesta para el equipo de rebalanceo, no una decisión del modelo.

### Consecuencias

- Una ventana puede cumplir las dos reglas a la vez (709 ventanas, 0.01%). Se le asigna la clase
  del umbral más cercano al inventario con que empieza la ventana: 664 de las 709 empiezan ya
  vacías o ya llenas, así que es la clase que ocurrió primero; en las otras 45 es una
  aproximación. 566 de las 709 son de la estación 51 (López Cotilla / Penitenciaría).
- Las cifras de esta decisión se reproducen en la sección 12 de
  `notebooks/2.0-rc-p2-tamano-ventana.ipynb`. Dos detalles de la simulación: el 50% de la
  capacidad no se redondea (con capacidad impar, las dos clases quedan a la misma distancia) y los
  movimientos del mismo segundo se ordenan de forma estable (primero las salidas), para que el
  resultado no dependa de la versión de pandas.
- Etiqueta final: 81.4% "sin riesgo", 9.8% "se vacía" y 8.8% "se llena", en 6,979,284 ventanas
  de 471 estaciones. El riesgo se concentra: la mitad de las estaciones pasa menos del 9% de sus
  ventanas en riesgo y el 10% pasa más de un tercio.
- La versión final está en `mibici_mlops/features.py`: `python -m mibici_mlops.features p2` guarda
  la etiqueta y la frecuencia de las últimas 4 semanas en `data/processed/p2_ventanas.parquet`. Da
  la misma etiqueta que el notebook en todas las ventanas.

## 6. Partición temporal y momento de predicción

- **Fecha:** 2026-10-17
- **Estado:** vigente
- **Aplica a:** el entrenamiento y la evaluación de P2 (baselines y candidatos).

### Contexto

Los modelos se evalúan con datos que no vieron al entrenar. Como P2 predice el futuro, la
partición no puede ser aleatoria: si una ventana de mañana queda en entrenamiento y la de hoy en
prueba, el modelo aprende con información del futuro. También hay que fijar en qué momento se
hace la predicción, porque las variables solo pueden usar lo que ya pasó a esa hora.

### Decisión

**Partición por fechas**, según el día de la simulación (de 4:00 a 3:59, decisión 4):

| Conjunto | Periodo | Meses | Ventanas | Se vacía | Se llena |
|---|---|---|---|---|---|
| Entrenamiento | 2022-03-01 a 2025-09-30 | 43 | 5,247,312 | 10.8% | 9.2% |
| Validación | 2025-10-01 a 2026-03-31 | 6 | 809,388 | 7.8% | 8.3% |
| Prueba | 2026-04-01 a 2026-09-30 | 6 | 922,584 | 5.7% | 7.2% |

**Momento de predicción:** cada día a las 4:00, para las 12 ventanas de ese día. Las variables
solo usan datos hasta las 3:59.

Las fechas están en `mibici_mlops/config.py` (`FECHA_INICIO`, `FECHA_INICIO_VALIDACION`,
`FECHA_INICIO_PRUEBA` y `FECHA_FIN`). Las cifras salen de la sección 13 de
`notebooks/2.0-rc-p2-tamano-ventana.ipynb`.

### Por qué

- **Por fechas y en orden:** los conjuntos no se mezclan y la prueba es lo más reciente, como
  cuando el modelo se use de verdad.
- **Validación y prueba juntas cubren un año completo** (octubre de 2025 a septiembre de 2026),
  con vacaciones de diciembre, Semana Santa y temporada de lluvias.
- **Septiembre de 2026 queda en prueba:** tiene 127% de los viajes de septiembre de 2019, el valor
  más alto de la comparación (decisión 2 de `decisiones.md`), pero los datos son reales y es lo
  que el modelo enfrentaría.
- **A las 4:00** ya se conoce todo el día anterior y todavía no empieza la operación. Coincide
  con el ML Canvas: corrida nocturna con datos hasta el cierre del día anterior.

### Hallazgos

- **El riesgo baja con el tiempo:** las clases de riesgo son el 20% de las ventanas de
  entrenamiento, el 16% de las de validación y el 13% de las de prueba.
- **Durante la prueba abren 98 estaciones** (mayo a agosto de 2026): son el 12% de las ventanas
  de prueba y casi nunca están en riesgo (~3%). Aun sin ellas, el riesgo en prueba (14%) es menor
  que en entrenamiento.

### Alternativas descartadas

- **Partición aleatoria:** mezcla días futuros en el entrenamiento.
- **Sacar de la prueba las estaciones nuevas:** la prueba se vería mejor, pero en uso real el
  modelo sí tendría que lidiar con ellas.
- **Validación cruzada temporal (varios cortes):** más robusta, pero más compleja; se puede usar
  al ajustar los candidatos.

### Consecuencias

- Los baselines no usan validación (no tienen nada que ajustar); los candidatos la usarán para
  elegir hiperparámetros.
- Los modelos se comparan siempre en el mismo conjunto. Las métricas de validación y de prueba no
  se comparan entre sí: la proporción de riesgo es distinta y la PR-AUC depende de ella.
- El baseline de frecuencia usa las 4 semanas anteriores aunque caigan en otro conjunto (por
  ejemplo, abril de 2026 usa semanas de marzo). No es fuga: a las 4:00 esas semanas ya pasaron.
- `FECHA_FIN` fija el periodo: si se descargan meses nuevos, los resultados no cambian hasta que
  se mueva esa fecha.

## 7. Evaluación y registro de los baselines en MLflow

- **Fecha:** 2026-10-17
- **Estado:** vigente
- **Aplica a:** los runs de P2 en MLflow.

### Contexto

La decisión 3 define los baselines y las métricas (recall y PR-AUC por clase de riesgo). Para
registrarlos en MLflow falta fijar cómo el baseline de frecuencia elige una clase, cómo se llaman
las métricas y qué se registra para poder repetir cada run.

### Decisión

- **Baseline de frecuencia:** para cada clase, la fracción de las últimas 4 semanas (misma
  estación, mismo día de la semana y misma ventana) en que hubo esa clase. Las semanas en que la
  estación todavía no existía cuentan como "sin riesgo". Predice la clase más frecuente; en
  empate gana, en este orden, "se vacía", "se llena" y "sin riesgo". Ejemplo: 2 semanas "se
  vacía" y 2 "sin riesgo" → "se vacía".
- **Métricas**, sobre el conjunto de prueba y para cada clase de riesgo:

  | Métrica | En scikit-learn | Nombres en MLflow |
  |---|---|---|
  | Recall | `recall_score` | `test_recall_se_vacia`, `test_recall_se_llena` |
  | Precisión | `precision_score` | `test_precision_se_vacia`, `test_precision_se_llena` |
  | PR-AUC | `average_precision_score` | `test_pr_auc_se_vacia`, `test_pr_auc_se_llena` |

- **Parámetros de cada run:** fechas de la partición, versión de los datos (el md5 de
  `data/raw.dvc`), semilla y el modelo con sus hiperparámetros.
- **Etiquetas:** equipo (`43`), pregunta (`P2`) y commit de Git.
- **Experimento:** `mibici-equipo43`, el mismo para P1, P2 y P3; semilla `43`. Ambos en
  `config.py`.

### Por qué

- **Empate a favor del riesgo:** el ML Canvas prioriza el recall de las clases de riesgo, porque
  es peor dejar a un usuario sin bicicleta o sin anclaje que mandar un camión de más. El orden
  entre "se vacía" y "se llena" es arbitrario, pero fijo.
- **Precisión además de recall:** un modelo que marca todo como riesgo tiene recall de 100% y no
  sirve; la precisión lo delata.
- **PR-AUC y no ROC-AUC:** con clases minoritarias, la ROC-AUC puede verse alta aunque el modelo
  dé muchas falsas alarmas; la PR-AUC se enfoca en la clase de riesgo. Además no depende de un
  umbral, así que compara baselines y candidatos con sus probabilidades.
- **md5 de DVC como versión de los datos:** DVC ya calcula el hash de `data/raw/` en
  `data/raw.dvc`; si cambia cualquier archivo, cambia el hash.
- **Semilla:** los baselines no tienen nada aleatorio, pero queda registrada para los candidatos.

### Alternativas descartadas

- **Exactitud (accuracy):** predecir siempre "sin riesgo" da 87% de exactitud en prueba sin
  detectar nada (decisión 1).
- **Manifiesto propio con hashes:** repetiría lo que ya hace DVC.

### Consecuencias

- Los dos baselines son dos runs del mismo experimento; sirven para la captura de comparación de
  runs.
- Los runs se ejecutan con todo en commit, para que el commit registrado sea el código que los
  produjo.
