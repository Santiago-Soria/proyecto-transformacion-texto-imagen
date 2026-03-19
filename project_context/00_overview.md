# Objetivo General y especificos del proyecto 

## Información General
* **Institución:** Escuela Superior de Cómputo (ESCOM)
* **Proyecto:** Transformación de textos depresivos en imágenes para clasificación emocional mediante visión por computacional.
* **Integrantes:** Equipo de 3 estudiantes.

## Objetivo Principal
Desarrollar y evaluar una metodología de transformación de textos depresivos y no depresivos en representaciones visuales abstractas, a partir de su representación vectorial (embeddings), con el fin de determinar mediante la clasificación con modelos de visión computacional si las características emocionales se conservan a través del cambio de modalidad (texto a imagen).

## Objetivos específicos:
1. Documentar el estado del arte en técnicas de detección de depresión en textos y modelos de visión computacional para el reconocimiento de patrones emocionales en imágenes abstractas. (terminado)
2. Implementar una línea base de clasificación de texto utilizando modelos transformers, para obtener las métricas de rendimiento (F1-score) que servirán como punto de comparación. (terminado)
3. Generar representaciones visuales abstractas a partir de los vectores de características (embeddings) del texto, mediante algoritmos de mapeo (ruido de Perlin, sistemas de reacción-difusión y atractores extraños) que codifiquen la semántica del texto original. (por hacer)
4. Entrenar y evaluar modelos de visión computacional (CNN o ViT) para la clasificación de las imágenes generadas, analizando su capacidad para discriminar entre categorías (depresivo o no depresivo). (por hacer)
5. Contrastar estadísticamente los resultados de la clasificación visual frente a la línea base textual para confirmar o refutar la capacidad de preservación de la carga emocional en el proceso de transformación multimodal. (por hacer)


## Alcances y limitaciones:
El proyecto no busca sustituir los diagnósticos clínicos ni superar a los métodos tradicionales, solo refutar o confirmar la hipótesis (determinar mediante la clasificación con modelos de visión computacional si las características emocionales se conservan a través del cambio de modalidad de texto a imagen), tampoco tiene como objetivo desarrollar un sistema/software/página/interfaz.

## Metodología:
Para el desarrollo de este proyecto se optó por una metodología cuantitativa y experimental. El proceso metodológico se estructuró en las siguientes fases secuenciales:
### Proceso metodológico:
1. Fundamentación y preparación de datos (terminada) - Tiene como objetivo establecer las bases teóricas y preparar el recurso más importante del proyecto: los datos.
2. Modelado y Establecimiento de la Línea Base (terminada) - Construir y evaluar un clasificador textual de alto rendimiento que funciona como punto de referencia para medir el éxito del clasificador visual, además de generar los embeddings que utiizarán los algoritmos de mapeo.
3. Diseño de la síntesis y modelado experimental (por hacer) - Esta es la fase central e innovadora del proyecto. Su objetivo es desarrollar el flujo de procesamiento para la transformación de texto a imagen, a partir de los embeddings.
4. Modelado en Visión por Computadora (por hacer) - Esta es la fase intermedia entre lo innovador y los resultados. Su objetivo es clasificar las imágenes generadas. 
5. Evaluación comparativa, análisis e informe de los resultados - La fase final se enfoca en validar la hipótesis central del proyecto e interpretar los resultados. 

### Marco experimental (arquitectura del pipeline):
El enfoque experimental se concibe como un pipeline de procesamiento de datos multimodal, diseñado con tres módulos principales que garantizan el flujo continuo de la información desde su entrada como texto hasta su clasificación como imagen.
1. **Módulo 1 - Procesamiento de Lenguaje Natural**
   - **Entrada:** Textos etiquetados como depresivos / no depresivos del conjunto de datos que realizó nuestro director de tesis.
   - **Proceso:** Preprocesamiento, desinfección ded sesgos, entrenamiento y extracción de embeddings con transformers (Se hizo una experimentación en donde se concluyó que el mejor fue 		BETO).
   - **Salida:** Embeddings de 768 dimensiones que condensa la representación semántica y emocional del texto analizado.

2. **Módulo 2 - Transformación de texto a imagen:**
   - **Entrada:** Embeddings de 768 dimensiones del módulo 1.
   - **Proceso:** Reducción de dimensionalidad de los embeddings con UMAP, parametrización de los componentes para cada algoritmo de mapeo (ruido de Perlin, atractores extraños, sistemas de reacción-difusión)
   - **Salida:** Imagenes generadas por los algoritmos de mapeo (se elegirá el mejor algoritmo de mapeo para crear un conjunto de datos con las imágenes que generó).

3. **Módulo de Visión por Computadora (Clasificación):**
   - **Entrada:** Imágenes generadas
   - **Proceso:** Extracción de características visuales y evaluación a través de un modelo de clasificación convolucional (CNN) o transformador visual (ViT).
   - **Salida:** Etiqueta de clasificación emocional final y porcentaje de confianza.