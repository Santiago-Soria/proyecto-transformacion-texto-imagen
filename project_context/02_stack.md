# Stack tecnológico:

## Lenguaje de programación: Python 3.x
## Entorno de desarrollo: Jupyter Notebooks, Google Colab (GPU NVIDIA Tesla T4), GitHub
## Repositorio: https://github.com/Santiago-Soria/proyecto-transformacion-texto-imagen
### Estructura del repositorio:
	proyecto-transformacion-texto-imagen/
	├── data/processed/         ← train.csv, validation.csv, test.csv
	├── data/shared/            ← umap_params.pkl (por generar)
	├── models/
	│   ├── best_model/         ← Exp_4.1_BETO_HPO_embeddings.pkl
	│   └── checkpoints/dccuchile_bert-base-spanish-wwm-cased/checkpoint-57/
	├── results/                ← comparacion_sesgo_pandemia.json
	├── src/
	│   ├── features/extraction.py
	│   ├── models/train_classic.py, train_transformer.py
	│   └── preprocessing_utils.py
	└── notebooks/
   	 ├── 2_run_experimentos_baseline.ipynb
   	 ├── 3_hyperparameter_tuning.ipynb
   	 └── 4_analisis_pandemia_bias.ipynb

## Frameworks y bibliotecas:
HuggingFace Transformers
HuggingFace Datasets
PyTorch
Optuna
scikit-learn
Polars
NumPy
Matplotlib
Seaborn

## Modelos pre-entrenados utilizados en el módulo 1:
BETO (modelo ganador de donde se extrajeron los embeddings)
RoBERTa-BNE
mDeBERTa-v3-base
XLM-RoBERTa-base

## Herramientas para los módulos siguientes:
- umap-learn          ← reducción de dimensionalidad
- noise / opensimplex ← Ruido de Perlin (no SciPy)
- Pillow (PIL)        ← manipulación y guardado de imágenes PNG
- numba / CuPy        ← aceleración para Gray-Scott (RD)
- scipy.integrate     ← solver ODE para Atractores (esto SÍ es SciPy)
- matplotlib          ← renderizado y colormaps
- CNN / ViT (por definir)


