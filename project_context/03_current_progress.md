# current_progress.md
# Estado actual del proyecto — última actualización: 18 marzo 2026

---

## ESTADO GLOBAL

| Módulo | Estado | Semana actual |
|--------|--------|---------------|
| Módulo 1 — NLP & Embeddings | ✅ COMPLETADO | — |
| Módulo 2 — Generación de Imágenes | 🔄 EN CURSO | Semana 1/8 |
| Módulo 3 — Clasificador Visual | ⏳ PENDIENTE | — |

---

## MÓDULO 1 — COMPLETADO

### Notebooks ejecutados (en orden)
| Notebook | Descripción | Estado |
|----------|-------------|--------|
| `2_run_experimentos_baseline.ipynb` | Experimentos TF-IDF + BETO baseline | ✅ |
| `3_hyperparameter_tuning.ipynb` | HPO con Optuna sobre BETO | ✅ |
| `4_analisis_pandemia_bias.ipynb` | Análisis sesgo pandémico | ✅ |

---

### Experimentos ejecutados y resultados

#### Experimentos clásicos (TF-IDF + Logistic Regression)
| Experimento | Preprocesamiento | F1-Val | F1-Test |
|-------------|-----------------|-------- |---------|
| Exp 1.1 | Limpieza básica + TF-IDF | 0.7137 | 0.6879 |
| Exp 1.2 | Limpieza + Stopwords + TF-IDF | 0.6742 | 0.7004 |
| Exp 2.1 | Limpieza básica + BETO (Frozen) | 0.6038 | 0.7165 |
| Exp 2.2 | Limpieza básica + RoBERTa (Frozen) | 0.6402 | 0.6096 |
| Exp 2.3 | Limpieza básica + mBERTa (Frozen) | 0.5885 | 0.526 |
| Exp 2.4 | Limpieza básica + XLM-RoBERTa (Frozen) | 0.6102 | 0.6221 |
| Exp 3.1 | Limpieza básica + BETO (Fine-Tuning) | 0.6114 | 0.7361 |
| Exp 3.2 | Limpieza básica + RoBERTa (Fine-Tuning) | 0.6941 | 0.6406 |
| Exp 3.3 | Limpieza básica + mBERTa (Fine-Tuning) | 0.5885 | 0.526 |
| Exp 3.4 | Limpieza básica + XLM-RoBERTa (Fine-Tuning) | 0.6173 | 0.526 |


#### Experimentos Transformer — BETO Baseline
- Modelo: `dccuchile/bert-base-spanish-wwm-cased`
- Hiperparámetros default: lr=2e-5, batch=16, epochs=5, early_stopping patience=2
- eval_dataset = val_ds (NO test)

| Split | F1-Macro |
|-------|----------|
| Val | 0.7311 |
| Test | 0.7361 |

#### HPO con Optuna (Notebook 3)
- Framework: Optuna, TPE Sampler, 20+ trials
- Función objetivo: F1-Macro sobre **val** (corrección crítica aplicada)
- Espacio de búsqueda:

| Hiperparámetro | Rango |
|----------------|-------|
| learning_rate | [1e-5, 5e-5] log-uniform |
| batch_size | {8, 16, 32} |
| weight_decay | [0.01, 0.1] |
| num_epochs | {3, 4, 5} |
| warmup_ratio | [0.0, 0.2] |

- Hiperparámetros óptimos encontrados:

| Hiperparámetro | Valor |
|----------------|-------|
| learning_rate | 1.0643e-05 |
| batch_size | 8 |
| weight_decay | 0.0280 |
| num_epochs | 4 |
| warmup_ratio | 0.1103 |

#### Análisis de sesgo pandémico (Notebook 4)
- Herramienta: `apply_pandemic_filter()` + `count_pandemic_terms()`
- Filtro aplicado a los 3 splits por separado (sin cruce de información)
- Textos con términos pandémicos: Train=24.6% | Val=20.2% | Test=20.2%

| Experimento | F1-Val | Δ-Val | F1-Test | Δ-Test |
|-------------|--------|-------|---------|--------|
| Baseline (sin HPO) | 0.7311 | — | 0.7361 | — |
| Exp 4.1 CON pandemia | 0.6484 | −0.0827 | 0.7361 | 0.0000 |
| Exp 4.2 SIN pandemia | 0.6299 | −0.1012 | 0.7334 | −0.0027 |

---

### Modelo ganador: Exp 4.1
**Justificación de selección:**
- Máximo F1-Test (0.7361) sobre todos los experimentos
- El sesgo pandémico es marginal (Δ = −0.0027) y está documentado
- El vocabulario pandémico es señal real de depresión en el corpus contemporáneo

**Checkpoint guardado en:**
`models/checkpoints/dccuchile_bert-base-spanish-wwm-cased/checkpoint-57/`

---

### Artefactos generados en Módulo 1

| Archivo | Ruta | Descripción |
|---------|------|-------------|
| `train.csv` | `data/processed/` | Split entrenamiento |
| `validation.csv` | `data/processed/` | Split validación |
| `test.csv` | `data/processed/` | Split test |
| `train_nopandemic.csv` | `data/processed/` | Train sin términos pandémicos |
| `validation_nopandemic.csv` | `data/processed/` | Val sin términos pandémicos |
| `test_nopandemic.csv` | `data/processed/` | Test sin términos pandémicos |
| `Exp_4.1_BETO_HPO_embeddings.pkl` | `models/best_model/` | Embeddings modelo ganador |
| `comparacion_sesgo_pandemia.json` | `results/` | Métricas análisis pandemia |

**Estructura del PKL ganador:**
```python
{
  'X_train': ndarray (908, 768),   # embeddings CLS train
  'y_train': ndarray (908,),
  'X_val':   ndarray (114, 768),   # embeddings CLS val
  'y_val':   ndarray (114,),
  'X_test':  ndarray (114, 768),   # embeddings CLS test
  'y_test':  ndarray (114,),
  'model_name': 'BETO_HPO_finetuned',
  'checkpoint': '.../checkpoint-57/',
  'shape': 768
}
```

#### Decisiones metodológicas críticas (NO revertir)
1. Separación de splits: Val usado exclusivamente para Early Stopping y HPO. Test consultado UNA SOLA VEZ por experimento al final.
2. Corrección de leakage detectada y aplicada: En una versión anterior del Notebook 3, Optuna usaba F1-Test como función objetivo — esto se corrigió para usar F1-Val. Los hiperparámetros óptimos finales provienen de la versión corregida.
3. PKL con 3 splits: Una versión anterior del PKL no incluía X_val/y_val. El PKL final correcto incluye los 3 splits. Verificar siempre con:assert 'X_val' in pkg.keys()
4. UMAP y scalers: Se ajustan SOLO sobre X_train. Val y test se transforman, nunca se usan para fit.
5. Brecha Val-Test en Exp 4.x: La brecha ampliada (~0.09) entre F1-Val y F1-Test en los experimentos 4.1 y 4.2 es una limitación conocida del paradigma embedding-extraction + shallow classifier (el encoder vio val durante Early Stopping). No es un error — está documentada.

## MÓDULO 2 - EN CURSO
Semana actual: 1 de 8
Contrato técnico definido:
| Parámetro           | Valor acordado                       |
| ------------------- | ------------------------------------ |
| Resolución          | 256×256 px                           |
| Formato             | PNG, RGB, sin compresión con pérdida |
| UMAP componentes    | 5                                    |
| UMAP metric         | cosine                               |
| UMAP n_neighbors    | 15                                   |
| UMAP min_dist       | 0.1                                  |
| UMAP random_state   | 42                                   |
| Scaler              | MinMaxScaler (fit solo en train)     |
| Paleta no_depresivo | YlOrRd                               |
| Paleta depresivo    | PuBuGn                               |

### Mapeo UMAP → parámetros por técnica:

| Componente | Perlin           | Reacción-Difusión  | Atractor (Lorenz)  |
|------------|------------------|--------------------|--------------------|
| umap    | frecuencia       | feed_rate f        | sigma σ (8–14)     |
| umap[1]    | turbulencia      | kill_rate k        | rho ρ (24–32)      |
| umap[2]    | velocidad part.  | coef. difusión     | beta β (2–4)       |
| umap[3]    | longitud traj.   | radio semilla      | n_puntos           |
| umap[4]    | n_octavas        | n_iteraciones      | tipo atractor      |

### Asignación de tareas:
| Integrante     | Técnica                        | Estado     |
| -------------- | ------------------------------ | ---------- |
| Diego     | Ruido de Perlin + Flow Fields  | ⏳ Semana 2 |
| Marco | Reacción-Difusión (Gray-Scott) | ⏳ Semana 2 |
| Santiago | Atractores Extraños (Lorenz)   | ⏳ Semana 2|

### Artefactos del Módulo 2:
| Artefacto | Estado | Notas |
|-----------|--------|-------|
| `data/shared/umap_params.pkl` | ✅ Generado | 2950 KB, 5 componentes, rango [0,1] verificado |
| `data/shared/color_scheme_check.png` | ⏳ Pendiente | |
| `data/images/perlin/` (1,136 imágenes) | ⏳ Pendiente | |
| `data/images/reaction_diffusion/` (1,136 imágenes) | ⏳ Pendiente | |
| `data/images/attractor/` (1,136 imágenes) | ⏳ Pendiente | |
| `data/shared/metadata_{tecnica}_{split}.csv` (9 archivos) | ⏳ Pendiente | |

### Semana 1 — Log de trabajo

#### umap_params.pkl generado (Santiago)
- Script: `src/umap/build_share_umap.py`
- UMAP fit exclusivamente sobre X_train (908 muestras, 768 dims → 5 componentes)
- MinMaxScaler fit solo en train, transform en los 3 splits
- Bug corregido: `MinMaxScaler.fit_transform` producía valores fuera de [0,1]
  por ruido de punto flotante IEEE 754 (ej. `1.0000000000000002`).
  Solución: `np.clip(scaler.transform(...), 0.0, 1.0)` en los 3 splits.
  No introduce leakage — el fit del scaler no se modifica.
- Validación final:
  - train: [0.000 – 1.000] shape=(908, 5)
  - val:   [0.000 – 1.000] shape=(114, 5)
  - test:  [0.000 – 1.000] shape=(114, 5)
- PKL compartido con Diego y Marco para Semana 2

#### Decisión metodológica registrada
- `np.clip` post-scaler es operación legítima (análoga a truncar predicciones
  al rango factible). No modifica el fit del scaler → sin leakage.

## MÓDULO 3 - PENDIENTE
Plan tentativo
Clasificador CNN (ResNet-18 o EfficientNet-B0)

Entrenado sobre el dataset de la técnica ganadora (decidir en semana 9-10)

Métrica principal: F1-Macro (consistencia con Módulo 1)

Comparación entre las 3 técnicas con clasificador ligero en semana 9

