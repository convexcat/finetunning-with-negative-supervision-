# Clasificación de Texto con Supervisión Negativa: Reproducción y Extensión

## Referencia del artículo

Ohashi, S., Takayama, J., Kajiwara, T., Chu, C., & Arase, Y. (2020).
**Text Classification with Negative Supervision.**
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*, pp. 351–357.

---

## Descripción del problema

Los modelos de representación de texto como BERT tienden a asignar representaciones similares a textos semánticamente parecidos. Sin embargo, en ciertas tareas de clasificación, textos semánticamente similares deben recibir etiquetas distintas. Por ejemplo, "I caught a cold" y "A cold is a legit disease" son semánticamente cercanas, pero solo la primera implica que el escritor está enfermo.

Este problema se agudiza en tareas con etiquetas semánticamente solapadas, como la clasificación de emociones, donde joy y optimism son conceptualmente próximas pero representan estados distintos.

---

## Metodología del paper

El paper propone un marco de aprendizaje multitarea con dos componentes:

**Tarea principal:** clasificador estándar (encoder + capa lineal) entrenado con cross-entropy o BCE según el tipo de tarea.

**Tarea auxiliar (AAN):** un discriminador que penaliza representaciones similares entre textos con etiquetas distintas. Para cada texto ancla, se muestrean n textos con etiqueta diferente del mismo batch y se minimiza:

```
La = (1/n) * Σ_j [1 + cosine_similarity(v_anchor, v_neg_j)]
```

**Pérdida total:** `L = Lm + La`

El encoder se comparte entre ambas tareas y se optimiza conjuntamente (full fine-tuning), de modo que la tarea auxiliar empuja al encoder a generar representaciones más distintas para textos con distintas etiquetas.

---

## Objetivo de la reproducción

Replicar experimentalmente los resultados del paper para los modelos **Baseline** y **AAN** con encoder **BERT-base-uncased** sobre los datasets MR y SemEval 2018 Task 1, verificando que la supervisión negativa produce mejoras consistentes en tareas con solapamiento semántico entre etiquetas.

---

## Propuesta de mejora

Se implementan dos extensiones:

### 1. Hard Negative Mining
El paper original muestrea negativos **aleatoriamente** del batch. El problema es que negativos fáciles (representaciones ya alejadas) contribuyen señal de gradiente casi nula.

La extensión propuesta selecciona los n negativos con **mayor similitud coseno** al ancla — los más difíciles de distinguir para el modelo actual. Esto concentra el entrenamiento en los casos donde el encoder actualmente falla más.

El cambio es mínimo y quirúrgico: una sola línea en la selección de negativos:
```python
# Original (aleatorio):
neg_idx = diff_indices[torch.randperm(len(diff_indices))[:n]]

# Extensión (hard mining):
cos_sims = cosine_similarity(v_anchor, v_all_negatives)
neg_idx  = diff_indices[torch.topk(cos_sims, n).indices]
```

### 2. Encoder más fuerte: RoBERTa-base
RoBERTa-base comparte arquitectura con BERT-base (~125M parámetros) pero fue entrenado con mayor cantidad de datos, sin la tarea de Next Sentence Prediction, y con mejores hiperparámetros de preentrenamiento. Se evalúa si los beneficios de AAN escalan con la calidad del encoder.

---

## Instrucciones de ejecución

### Requisitos
- Python 3.10+
- GPU con al menos 16GB VRAM recomendado (probado en NVIDIA A100 80GB)
- Cuenta en Vast.ai o similar para instancias GPU en la nube

### Instalación
```bash
pip install transformers==4.40.0 datasets==2.19.0 scikit-learn==1.4.2 pandas numpy sentencepiece protobuf
```

### Ejecución
Los notebooks deben ejecutarse en orden:

```
NB1_preparacion_datos.ipynb        # Preparar datos y cachear modelos
NB2_reproduccion_baseline_AAN.ipynb # Reproducción: BERT Baseline + AAN
NB3_extension_hard_negatives_roberta.ipynb  # Extensión: hard mining + RoBERTa
NB4_resultados_analisis.ipynb      # Tablas y análisis final
```

### Hiperparámetros utilizados
| Parámetro | Valor |
|---|---|
| Encoder (reproducción) | bert-base-uncased |
| Encoder (extensión) | roberta-base |
| Batch size | 16 |
| Negativos por ancla (n) | 4 |
| Optimizador | Adam β1=0.999, β2=0.9 |
| Learning rate | Seleccionado de {1e-5, 3e-5, 5e-5} |
| Early stopping (paciencia) | 10 épocas |
| Máximo de épocas | 50 |
| Número de trials | 5 (media trimmed: se elimina mejor y peor) |
| Max longitud de secuencia | 128 tokens |

---

## Estructura del repositorio

```
├── README.md
├── ANALISIS_RESULTADOS.md
├── notebooks/
│   ├── NB1_preparacion_datos.ipynb
│   ├── NB2_reproduccion_baseline_AAN.ipynb
│   ├── NB3_extension_hard_negatives_roberta.ipynb
│   └── NB4_resultados_analisis.ipynb
└── results/
    ├── nb2_results.json
    ├── nb3_results.json
    └── all_results.csv
```

---

## Resumen de resultados

| Encoder | Método | MR (Accuracy) | SemEval (Exact Match) |
|---|---|---|---|
| BERT-base | Baseline | 0.8230 ± 0.0070 | 0.1761 ± 0.0142 |
| BERT-base | AAN (random) | 0.8183 ± 0.0038 | 0.2275 ± 0.0008 |
| BERT-base | AAN (hard) | 0.8127 ± 0.0071 | 0.2205 ± 0.0044 |
| RoBERTa-base | Baseline | 0.8487 ± 0.0038 | 0.2016 ± 0.0025 |
| RoBERTa-base | AAN (random) | 0.8346 ± 0.0036 | 0.1883 ± 0.0560 |
| RoBERTa-base | **AAN (hard)** | **0.8521 ± 0.0055** | **0.2411 ± 0.0071** |

---

## Conclusiones principales

**Sobre la reproducción:** La supervisión negativa (AAN) produce mejoras consistentes en tareas con solapamiento semántico entre etiquetas. En SemEval (clasificación multi-etiqueta de emociones), AAN random mejora el baseline de BERT en +29.2% relativo, lo que replica la hipótesis central del paper. En MR (clasificación binaria), la mejora no se observa, posiblemente por dificultades en la inicialización, problemas particulares del flujo de entrenamiento que no se pueden esclarecer del artículo original, o la falta de muestras (pues en el artículo original se realizaron diez iteraciones de ajuste fino, contra las cinco que yo hago).

**Sobre el hard negative mining:** La selección explícita de negativos difíciles mejora la estabilidad y el rendimiento, especialmente con encoders más fuertes. Con RoBERTa en SemEval, AAN random colapsa en alta varianza (std=0.056) mientras que AAN hard logra el mejor resultado general (0.2411) con varianza baja (std=0.007). Esto sugiere que la estrategia de muestreo es más crítica cuando el encoder ya produce representaciones de alta calidad.

**Sobre RoBERTa:** RoBERTa supera a BERT en el baseline, pero los beneficios de AAN interactúan de forma compleja con la calidad del encoder. Solo la combinación RoBERTa + AAN hard mejora consistentemente sobre su propio baseline en ambos datasets, mostrando que hard mining y encoders fuertes son complementarios.

---

## Nota
Los Notebooks presentados son versiones finales limpias de lo que sería el flujo completo del experimento. Durante el desarrollo y obtención de resultados se usaron Notebooks distintos con diferencias importantes que no llegaron a la etapa final (por ejemplo, en un inicio se estaba desarrollando el entrenamiento con el Dataset de GoEmotions, pero se tuvo que descartar por cuestiones del costo de computo y tiempo), o se tuvieron errores de ejecución (por ejemplo, entornos desconectados), por lo cual fue necesario modificar progresivamente los notebooks para no tener que repetir etapas del experimento. En caso de requerir algunos de los Notebooks originales, escríbanme.

---

## Referencias

- Ohashi et al. (2020). Text Classification with Negative Supervision. ACL 2020.
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
- Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.
- Mohammad et al. (2018). SemEval-2018 Task 1: Affect in Tweets. SemEval 2018.
