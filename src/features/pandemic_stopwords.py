"""
Módulo para el tratamiento del sesgo de términos de pandemia en el dataset.
Los términos fueron identificados mediante análisis de frecuencia por clase
(ver reports/analisis_pandemia_bias.png).
"""

import re
from typing import List

# ============================================================
# TÉRMINOS DE PANDEMIA IDENTIFICADOS EN EL ANÁLISIS
# Fuente: Gráfica de frecuencia por clasificación
# ============================================================
PANDEMIC_TERMS = [
    # Términos directos de pandemia (alta frecuencia en "No Depresivo")
    "pandemia", "cuarentena", "covid", "virus", "confinamiento",
    "encierro", "contingencia",

    # Variantes y términos relacionados
    "coronavirus", "sars", "covd", "covid19", "covid-19",
    "aislamiento", "distanciamiento", "sanitario", "sanitaria",
    "mascarilla", "cubrebocas", "vacuna", "vacunacion",
    "epidemia", "brote", "contagio", "contagios", "infectado",
    "positivo", "negativo",  # en contexto médico
    "hospital", "icu", "uci", "intubado", "ventilador",
    "nueva normalidad", "fase", "semaforo",
    "sana distancia", "quedateencasa", "quédate en casa",

    # Términos institucionales del contexto mexicano
    "ssa", "imss", "issste", "secretaria de salud",
]

# Compilar regex una sola vez para eficiencia
_PANDEMIC_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in PANDEMIC_TERMS) + r')\b',
    flags=re.IGNORECASE
)


def remove_pandemic_terms(text: str) -> str:
    """
    Elimina los términos de pandemia de un texto, tratándolos como stopwords.

    Args:
        text: Texto original a limpiar.

    Returns:
        Texto sin términos de pandemia, con espacios normalizados.
    """
    cleaned = _PANDEMIC_PATTERN.sub('', text)
    # Normalizar espacios múltiples generados por la eliminación
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def apply_pandemic_filter(texts: List[str]) -> List[str]:
    """
    Aplica remove_pandemic_terms a una lista de textos.

    Args:
        texts: Lista de textos originales.

    Returns:
        Lista de textos sin términos de pandemia.
    """
    return [remove_pandemic_terms(t) for t in texts]


def count_pandemic_terms(text: str) -> int:
    """
    Cuenta cuántos términos de pandemia contiene un texto.
    Útil para análisis estadístico del sesgo.

    Args:
        text: Texto a analizar.

    Returns:
        Número de términos de pandemia encontrados.
    """
    return len(_PANDEMIC_PATTERN.findall(text))
