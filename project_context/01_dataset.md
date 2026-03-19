# Conjunto de datos utilizado

🧠 Conjunto de datos de textos en español para la detección de la depresión

Conjunto de datos asociado al artículo:

**SpADE-BERT: Multilingual BERT-based model with trigram-sensitive tokenization, tuned for depression detection in Spanish texts**

📄 Articulo:
[https://www.mdpi.com/2673-2688/7/2/48](https://www.mdpi.com/2673-2688/7/2/48)

---

# 📄 Descripción del conjunto de datos:

Este conjunto de datos contiene textos escritos por personas que padecen depresión. El conjunto de datos fue validado por psicólogos expertos y se elaboró ​​mediante un proceso de verificación en dos etapas.

### Proceso de validación

1. Aplicación del **Inventario de Depresión de Beck (BDI)**.

2. Validación independiente por **al menos tres psicólogos**.

El conjunto de datos está destinado a la investigación en **Procesamiento del Lenguaje Natural (PLN)** y **Aprendizaje Automático** para el análisis de la salud mental.

---

# 📊 Estructura

**Total records:** 1,136
**Format:** CSV
**Lenguaje:** Español

### Columns

| Columna                | Descripción                     |
| --------------------- | -------------------------------- |
| id                    | Identificador de registro        |
| personal_key          | Identificador anónimo            |
| depression_level      | Resultado del Inventario de Depresión de Beck |
| text                  | Texto escrito por el individuo   |
| manual_classification | Validación del psicólogo         |
| age                   | Edad                             |
| occupation            | Ocupación                        |
| max_level_studies     | Máximo nivel de estudios         |
| residence             | Lugar de residencia              |
| marital_status        | Estado civil                     |

---

# División estratificada el conjunto de datos:
- Splits: Train=908 | Val=114 | Test=114
- Distribución train: Dep=353, No_dep=555