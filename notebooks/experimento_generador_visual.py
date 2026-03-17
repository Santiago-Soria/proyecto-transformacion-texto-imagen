import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from typing import List

def generar_atractor_clifford(vector_umap: List[float], iteraciones: int = 1000000):
    """
    Genera un Atractor de Clifford mapeando las 4 dimensiones de UMAP a los parámetros a, b, c, d.
    """
    print("Generando Atractor de Clifford...")
    a, b, c, d = vector_umap
    
    xs = np.zeros(iteraciones)
    ys = np.zeros(iteraciones)
    xs[0], ys[0] = 0.1, 0.1
    
    for i in range(iteraciones - 1):
        xs[i+1] = np.sin(a * ys[i]) + c * np.cos(a * xs[i])
        ys[i+1] = np.sin(b * xs[i]) + d * np.cos(b * ys[i])
        
    plt.figure(figsize=(8, 8), facecolor='black')
    plt.plot(xs, ys, ',', color='cyan', alpha=0.05) 
    plt.axis('off')
    plt.title(f"Atractor de Clifford (Params: {a:.2f}, {b:.2f}, {c:.2f}, {d:.2f})", color='white')
    plt.show()

def generar_mancha_abstracta(vector_umap: List[float], resolucion: int = 150):
    """
    Genera una textura abstracta usando PerlinNoise y el vector de 4 dimensiones.
    Resolución bajada a 150 para que renderice rápido en la prueba.
    """
    print("Generando Mancha Abstracta (Ruido de Perlin)...")
    
    # 1. Mapeo Semántico-Visual
    escala = abs(vector_umap[0]) + 0.5          # Controla el "zoom" 
    octavas = int(abs(vector_umap[1]) * 4) + 1  # Controla el detalle (1 a 5)
    color_hue = vector_umap[2]                  # Desplazamiento de color
    contraste = vector_umap[3]                  # Multiplicador de contraste
    
    # Inicializamos el generador de ruido con las octavas calculadas
    noise = PerlinNoise(octaves=octavas, seed=42)
    imagen = np.zeros((resolucion, resolucion))
    
    # 2. Generación del campo de ruido
    for i in range(resolucion):
        for j in range(resolucion):
            # Normalizamos coordenadas
            x = i / resolucion * escala
            y = j / resolucion * escala
            
            # Generamos el ruido y aplicamos el contraste/hue
            valor_ruido = noise([x, y])
            imagen[i, j] = (valor_ruido * contraste) + color_hue
            
    # 3. Renderizado
    plt.figure(figsize=(8, 8))
    plt.imshow(imagen, cmap='magma', interpolation='bicubic')
    plt.axis('off')
    plt.title("Mapeo Abstracto Perlin", color='black')
    plt.show()

if __name__ == "__main__":
    # Vector simulado de BETO + UMAP
    vector_depresivo_simulado = [1.5, -1.2, 1.8, -0.9]
    
    generar_atractor_clifford(vector_depresivo_simulado)
    generar_mancha_abstracta(vector_depresivo_simulado)