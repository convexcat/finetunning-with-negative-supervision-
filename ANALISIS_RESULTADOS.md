# Análisis de Resultados

## Resumen de experimentos

Se entrenaron 6 configuraciones de modelos sobre dos datasets: MR (clasificación binaria de sentimiento) y SemEval 2018 Task 1 Emotions (clasificación multi-etiqueta de emociones). Todos los modelos usan full fine-tuning — todos los parámetros del encoder y la cabeza de clasificación se optimizan conjuntamente. Los resultados reportados son la media trimmed de 5 trials independientes (se elimina el mejor y el peor).

| Encoder | Método | MR (Accuracy) | SemEval (Exact Match) |
|---|---|---|---|
| BERT-base | Baseline | 0.8230 ± 0.0070 | 0.1761 ± 0.0142 |
| BERT-base | AAN random | 0.8183 ± 0.0038 | 0.2275 ± 0.0008 |
| BERT-base | AAN hard | 0.8127 ± 0.0071 | 0.2205 ± 0.0044 |
| RoBERTa-base | Baseline | 0.8487 ± 0.0038 | 0.2016 ± 0.0025 |
| RoBERTa-base | AAN random | 0.8346 ± 0.0036 | 0.1883 ± 0.0560 |
| RoBERTa-base | AAN hard | **0.8521 ± 0.0055** | **0.2411 ± 0.0071** |

---

## Hallazgo 1: La supervisión negativa ayuda principalmente en tareas con solapamiento semántico entre etiquetas

En MR, tarea binaria simple donde las clases positivo/negativo son semánticamente distintas, la supervisión negativa no aportó mejora con BERT. AAN random (0.8183) y AAN hard (0.8127) quedan por debajo del baseline (0.8230). Esto sugiere que cuando el encoder ya separa bien las clases, la tarea auxiliar introduce ruido más que señal útil.

En SemEval, la situación es opuesta. Las 11 etiquetas emocionales tienen un alto grado de solapamiento semántico — textos que expresan joy y optimism son lingüísticamente similares pero llevan etiquetas distintas. Aquí AAN random mejora el baseline en +0.0514 (+29.2% relativo), confirmando la hipótesis central del paper: la supervisión negativa es más valiosa cuando la tarea de clasificación es adversarial al encoder, es decir, cuando textos semánticamente similares deben recibir etiquetas diferentes.

Este hallazgo replica y extiende el resultado del paper original, que también observó mayores ganancias en datasets con etiquetas más ambiguas.

---

## Hallazgo 2: El hard negative mining es más robusto pero no uniformemente superior

Con BERT en MR, AAN hard (0.8127) queda ligeramente por debajo de AAN random (0.8183). En una tarea binaria con batch size 16, la mitad del batch es negativo por definición, por lo que el pool de negativos disponibles es pequeño. En este escenario, el hard mining no tiene suficiente variedad para seleccionar negativos genuinamente informativos y puede sobreenfocarse en casos ruidosos.

Con RoBERTa en SemEval, el hard mining produce el mejor resultado general del experimento (0.2411), superando tanto al baseline (0.2016, +19.6%) como a AAN random (0.1883, +28.0%). La diferencia entre hard y random es especialmente pronunciada en términos de estabilidad: AAN random con RoBERTa tiene una desviación estándar de 0.0560 — extraordinariamente alta — mientras que AAN hard tiene una desviación de apenas 0.0071. Esto indica que el muestreo aleatorio con un encoder más fuerte (que ya tiene representaciones más separadas desde el inicio) produce negativos inconsistentes que llevan el entrenamiento en direcciones impredecibles. El hard mining resuelve esto concentrando la señal en los casos que el modelo realmente necesita trabajar.

---

## Hallazgo 3: La interacción entre encoder y estrategia de muestreo es compleja

Un resultado contraintuitivo es que RoBERTa + AAN random (0.1883) queda por debajo de BERT + AAN random (0.2275) en SemEval. Esto es inesperado dado que RoBERTa es un encoder más fuerte.

Una explicación plausible es que RoBERTa, con representaciones de mayor calidad desde el inicio, produce negativos que ya están relativamente separados en el espacio de representaciones. Al muestrear aleatoriamente, muchos negativos seleccionados son "fáciles" — tienen baja similitud coseno con el ancla — y por tanto contribuyen gradientes casi nulos. Esto hace el entrenamiento ineficiente y variable. El hard mining corrige exactamente este problema: al seleccionar explícitamente los negativos más difíciles, garantiza que la tarea auxiliar siempre esté trabajando en casos informativos.

Esta interacción sugiere que la estrategia de muestreo de negativos es más crítica con encoders fuertes que con encoders débiles, un resultado que tiene implicaciones prácticas para la aplicación del método a modelos más grandes.

---

## Hallazgo 4: RoBERTa + AAN hard es la mejor configuración global

RoBERTa + AAN hard logra el mejor resultado en ambos datasets (0.8521 en MR, 0.2411 en SemEval). Más importante, es la única configuración que supera consistentemente su propio baseline en ambas tareas: +0.0034 en MR y +0.0395 en SemEval. Esto sugiere que la combinación de un encoder preentrenado de calidad con una selección informativa de negativos es la estrategia más robusta.

---

## Discrepancias con el paper original

El paper original reporta mejoras consistentes de AAN sobre el baseline en todos los datasets evaluados con BERT-base. En nuestra reproducción, AAN random con BERT no mejora el baseline en MR (0.8183 vs 0.8230). Posibles explicaciones:

- **Diferencia de dataset**: el paper usa el dataset MR de SentEval directamente; nosotros usamos la versión `rotten_tomatoes` de HuggingFace, que puede tener pequeñas diferencias en las particiones.
- **Reducción de trials**: el paper usa 10 trials con media trimmed de 8; nosotros usamos 5 trials con media de 3, lo que introduce más varianza estadística.
- **Diferencias de implementación menores**: tamaño de warmup del scheduler, inicialización de pesos de la cabeza de clasificación.

La dirección general de los resultados es consistente con el paper: AAN ayuda más en tareas con mayor solapamiento entre etiquetas. La diferencia en MR puede atribuirse a la simplicidad de la tarea y a las diferencias metodológicas menores descritas.

---

## Limitaciones

- Los experimentos se realizaron exclusivamente en inglés; el paper original incluye evaluación multilingüe (japonés y chino) que no se reprodujo.
- No se evaluó clasificación a nivel de documento (dataset arXiv del paper original) debido a restricciones computacionales.
- El barrido de hiperparámetros (λ y n) se omitió por restricciones de tiempo; se usaron los valores del paper (λ=1, n=4).
- Con solo 5 trials, los intervalos de confianza son más amplios que los reportados en el paper original.
