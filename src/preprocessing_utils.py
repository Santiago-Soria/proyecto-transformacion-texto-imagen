"""
Módulo de preprocesamiento de texto para clasificación de depresión.
Implementa tres niveles de limpieza: P1 (Mínima), P2 (Agresiva), P3 (Clásica).
"""

import re
import logging
from typing import Optional, Dict, List
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar modelo de español
try:
    nlp = spacy.load("es_core_news_sm")
    logger.info("Modelo spaCy cargado correctamente")
except OSError:
    logger.warning("Modelo spaCy no encontrado. Instalando...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load("es_core_news_sm")

# --- Configuración Inicial (Solo se ejecuta una vez) ---
def _download_resources():
    """Descarga recursos necesarios de NLTK."""
    resources = {
        'stopwords': 'corpus/stopwords',
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab'
    }
    
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Descargando recurso NLTK: {name}")
            nltk.download(name, quiet=True)

_download_resources()

# --- Personalización de Stop Words ---
# Basado en investigación: los pronombres de primera persona y negaciones
# son marcadores lingüísticos clave en detección de depresión
spanish_stopwords = set(stopwords.words('spanish'))

palabras_a_conservar = {
    # Pronombres de primera persona singular y plural 
    'yo', 'me', 'mi', 'mí', 'mío', 'mía', 'míos', 'mías', 'nosotros','nosotras', 'nuestro', 'nuestra', 'conmigo',
    # Palabras absolutistas
    'nada', 'todo', 'todos', 'mucho', 'muchos', 'poco', 'tanto','siempre', 'muy',
    # Palabras de negación
    'no','sin', 'ni', 'nunca',
    # Palabras relacionadas con emociones
    'siente', 'sentido', 'sentidos', 'sentidas', 'sentida', 'sentid', 
    # Palabras de analisis contextual
    'tú', 'ti', 'te', 'tu', 'tus', 'tuyo', 'tuya', 'tuyos', 'tuyas',
    'él', 'ella', 'ellos', 'ellas', 'le', 'les', 'lo', 'los', 'la', 'las', 'se',
    'sola', 'solo',  'pero', 'más', 'porque', 'contra', 'también', 'sí'
}

# Eliminamos las palabras clave de nuestra lista de stop words
STOPWORDS_PERSONALIZADAS = spanish_stopwords - palabras_a_conservar

# --- Funciones Auxiliares ---

def normalize_elongations(text: str) -> str:
    """
    Normaliza elongaciones emocionales (ej: 'muuuuy' -> 'muy').
    Las elongaciones son marcadores de intensidad emocional.
    
    Args:
        text: Texto a normalizar
        
    Returns:
        Texto con elongaciones normalizadas
    """
    # Reemplaza 3 o más repeticiones de una letra por 2
    # Ej: "noooooo" -> "noo", "muuuuy" -> "muy"
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def validate_text(text: str, min_length: int = 3) -> bool:
    """
    Valida que el texto procesado tenga contenido significativo.
    
    Args:
        text: Texto a validar
        min_length: Longitud mínima en caracteres
        
    Returns:
        True si el texto es válido, False en caso contrario
    """
    if not text or len(text.strip()) < min_length:
        return False
    # Verificar que tenga al menos una palabra alfabética
    if not re.search(r'[a-záéíóúñü]', text):
        return False
    return True

def get_preprocessing_stats(original: str, processed: str) -> Dict[str, int]:
    """
    Genera estadísticas del preprocesamiento.
    
    Args:
        original: Texto original
        processed: Texto procesado
        
    Returns:
        Diccionario con estadísticas
    """
    return {
        'original_length': len(original),
        'processed_length': len(processed),
        'original_words': len(original.split()),
        'processed_words': len(processed.split()),
        'reduction_percentage': round((1 - len(processed) / max(len(original), 1)) * 100, 2)
    }

# --- Definición de Pipelines de Limpieza ---

def clean_text_p1_minimal(texto: str, return_stats: bool = False) -> str:
    """
    P1: Limpieza Mínima (Baseline).
    
    Operaciones:
    - Conversión a minúsculas
    - Normalización de elongaciones emocionales
    - Eliminación de caracteres especiales (preserva acentos y ñ)
    - Normalización de espacios en blanco
    
    Args:
        texto: Texto a limpiar
        return_stats: Si True, retorna tupla (texto_limpio, estadísticas)
        
    Returns:
        Texto limpio o tupla (texto_limpio, stats) si return_stats=True
        
    Examples:
        >>> clean_text_p1_minimal("Me siento muuuuy triste 😢")
        'me siento muy triste triste'
    """
    if not isinstance(texto, str):
        logger.warning(f"Input no es string: {type(texto)}")
        return "" if not return_stats else ("", {})
    
    original = texto
    
    # Paso 1: Minúsculas
    texto = texto.lower()
    # Paso 2: Normalizar elongaciones emocionales
    texto = normalize_elongations(texto)
    # Paso 3: Eliminar caracteres especiales (preservar letras, números, acentos y ñ)
    texto = re.sub(r'[^a-z0-9áéíóúñü\s]', ' ', texto)
    # Paso 4: Normalizar espacios
    texto = re.sub(r'\s+', ' ', texto).strip()

    # Validación
    if not validate_text(texto):
        logger.warning(f"Texto resultante vacío o inválido después de P1. Original: {original[:50]}")
    
    if return_stats:
        stats = get_preprocessing_stats(original, texto)
        return texto, stats
    
    return texto


def clean_text_p2_aggressive(texto: str, return_stats: bool = False) -> str:
    """
    P2: Limpieza Agresiva.
    
    Operaciones:
    - Todas las operaciones de P1
    - Eliminación de stop words personalizadas (preservando marcadores de depresión)
    
    Args:
        texto: Texto a limpiar
        return_stats: Si True, retorna tupla (texto_limpio, estadísticas)
        
    Returns:
        Texto limpio o tupla (texto_limpio, stats) si return_stats=True
        
    Examples:
        >>> clean_text_p2_aggressive("Yo no me siento bien con esto")
        'yo no me siento bien'  # 'con' y 'esto' son stopwords
    """
    # Aplicar limpieza mínima
    if return_stats:
        texto_limpio, stats_p1 = clean_text_p1_minimal(texto, return_stats=True)
        original = texto_limpio
    else:
        texto_limpio = clean_text_p1_minimal(texto)
        original = texto_limpio
    
    if not texto_limpio:
        return ("", {}) if return_stats else ""
    
    # Tokenización
    try:
        palabras = word_tokenize(texto_limpio, language='spanish')
    except Exception as e:
        logger.error(f"Error en tokenización: {e}")
        palabras = texto_limpio.split()
    
    # Filtrar stop words personalizadas
    palabras_filtradas = [
        palabra for palabra in palabras 
        if palabra not in STOPWORDS_PERSONALIZADAS and len(palabra) > 1
    ]
    
    texto_final = " ".join(palabras_filtradas)
    
    # Validación
    if not validate_text(texto_final):
        logger.warning(f"Texto vacío después de P2. Original: {texto[:50]}")
    
    if return_stats:
        stats = get_preprocessing_stats(original, texto_final)
        stats['words_removed'] = len(palabras) - len(palabras_filtradas)
        return texto_final, stats
    
    return texto_final

def clean_text_p3_classic(texto: str, return_stats: bool = False) -> str:
    """
    P3: Limpieza Clásica con Lematizazción.
    
    Operaciones:
    - Todas las operaciones de P2
    - Lematización con spaCy
    
    NOTA: El stemming reduce palabras a su raíz lingüística, lo cual puede
    ser beneficioso para TF-IDF pero puede perder información contextual
    importante para transformers.
    
    Args:
        texto: Texto a limpiar
        return_stats: Si True, retorna tupla (texto_limpio, estadísticas)
        
    Returns:
        Texto limpio o tupla (texto_limpio, stats) si return_stats=True
    """
    # Aplicar limpieza agresiva
    if return_stats:
        texto_agresivo, stats_p2 = clean_text_p2_aggressive(texto, return_stats=True)
        original = texto_agresivo
    else:
        texto_agresivo = clean_text_p2_aggressive(texto, return_stats=False)
        original = texto_agresivo
    
    if not texto_agresivo:
        return ("", {}) if return_stats else ""
    
    # Lematización con spaCy
    try:
        doc = nlp(texto_agresivo)
        # Extraer lemmas, excluyendo puntuación y espacios
        lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        texto_final = " ".join(lemmas)
    except Exception as e:
        logger.error(f"Error en lematización con spaCy: {e}")
        # Fallback a texto sin lematizar
        texto_final = texto_agresivo
    
    if return_stats:
        stats = get_preprocessing_stats(original, texto_final)
        stats['method'] = 'lemmatization'
        return texto_final, stats
    
    return texto_final


# --- Función Unificada para Experimentación ---

def preprocess_text(texto: str, method: str = 'P1', return_stats: bool = False):
    """
    Función unificada para aplicar cualquier método de preprocesamiento.
    
    Args:
        texto: Texto a procesar
        method: Método a aplicar ('P1', 'P2', o 'P3')
        return_stats: Si True, retorna también estadísticas
        
    Returns:
        Texto procesado o tupla (texto, stats)
        
    Raises:
        ValueError: Si el método especificado no existe
    """
    method = method.upper()
    
    preprocessing_functions = {
        'P1': clean_text_p1_minimal,
        'P2': clean_text_p2_aggressive,
        'P3': clean_text_p3_classic
    }
    
    if method not in preprocessing_functions:
        raise ValueError(f"Método '{method}' no válido. Usa 'P1', 'P2', o 'P3'")
    
    return preprocessing_functions[method](texto, return_stats=return_stats)


# --- Funciones de Análisis ---

def compare_preprocessing_methods(texto: str) -> Dict[str, Dict]:
    """
    Compara los tres métodos de preprocesamiento en un texto.
    
    Args:
        texto: Texto a analizar
        
    Returns:
        Diccionario con resultados de cada método
        
    Examples:
        >>> results = compare_preprocessing_methods("Me siento muy triste y solo")
        >>> print(results['P1']['processed'])
    """
    results = {}
    
    for method in ['P1', 'P2', 'P3']:
        processed, stats = preprocess_text(texto, method=method, return_stats=True)
        results[method] = {
            'processed': processed,
            'stats': stats
        }
    
    return results


def batch_preprocess(texts: List[str], method: str = 'P1', 
                     show_progress: bool = True) -> List[str]:
    """
    Procesa una lista de textos con el método especificado.
    
    Args:
        texts: Lista de textos a procesar
        method: Método de preprocesamiento ('P1', 'P2', o 'P3')
        show_progress: Mostrar barra de progreso
        
    Returns:
        Lista de textos procesados
    """
    processed_texts = []
    total = len(texts)
    
    for i, text in enumerate(texts):
        if show_progress and i % 100 == 0:
            logger.info(f"Procesando: {i}/{total} ({i/total*100:.1f}%)")
        
        processed = preprocess_text(text, method=method)
        processed_texts.append(processed)
    
    if show_progress:
        logger.info(f"Completado: {total}/{total} (100%)")
    
    return processed_texts