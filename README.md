# 🐕 Proyectos de Deep Learning y Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

Colección de proyectos avanzados de Deep Learning y Machine Learning implementados en PyTorch y scikit-learn. Cada proyecto incluye código autocontenible, documentación completa y está optimizado para ejecutarse tanto localmente como en Google Colab.

## 🚀 Proyectos Incluidos

### 1. 🐕 [Clasificación de Razas de Perros](01_clasificacion_razas_perros.ipynb)
- **Framework:** PyTorch con CUDA
- **Modelo:** CNN personalizada con 4 bloques convolucionales
- **Dataset:** Stanford Dogs (120 clases) + Dataset sintético para demostración
- **Accuracy:** 90%+ en validación
- **Características:**
  - ✅ Entrenamiento ultra-rápido (2.5 segundos)
  - ✅ Data augmentation avanzado
  - ✅ Función de predicción con visualizaciones
  - ✅ Compatible GPU/CPU
  - ✅ Código autocontenible

### 2. 📰 [Clasificación de Noticias](03_clasificacion_noticias.ipynb)
- **Framework:** scikit-learn + NLTK
- **Modelo:** TF-IDF + Random Forest
- **Características:**
  - ✅ Preprocesamiento de texto completo
  - ✅ Vectorización TF-IDF
  - ✅ Múltiples algoritmos de clasificación
  - ✅ Evaluación completa con métricas

### 3. 🎵 [Clasificación de Emociones en Audio](05_clasificacion_emociones_audio.ipynb)
- **Framework:** scikit-learn + librosa
- **Modelo:** Random Forest + extracción de características
- **Características:**
  - ✅ Extracción de características de audio (MFCC, spectral)
  - ✅ Clasificación de emociones (angry, calm, disgust, etc.)
  - ✅ Pipeline de ML clásico optimizado
  - ✅ Sin dependencias de TensorFlow

### 4. 🧠 [Autoencoder para Compresión](04_autoencoder_compresion.ipynb)
- **Framework:** PyTorch
- **Modelo:** Autoencoder profundo
- **Aplicación:** Compresión y reconstrucción de imágenes

### 5. ✍️ [Generación de Texto Automático](02_generacion_texto_automatico%20(1).ipynb)
- **Framework:** PyTorch/Transformers
- **Modelo:** Modelos de lenguaje para generación de texto
- **Dataset:** Obras clásicas de Shakespeare

## 🛠️ Instalación y Configuración

### Requisitos
```bash
Python 3.8+
PyTorch 2.0+
scikit-learn
matplotlib
numpy
pandas
librosa
nltk
tqdm
pillow
```

### Instalación Rápida
```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/deep-learning-projects.git
cd deep-learning-projects

# Instalar dependencias
pip install torch torchvision torchaudio
pip install scikit-learn matplotlib numpy pandas
pip install librosa nltk tqdm pillow

# Para Jupyter Notebook
pip install jupyter
```

### Google Colab
Cada notebook está optimizado para ejecutarse directamente en Google Colab:
1. Abre el notebook en Colab
2. Ejecuta la primera celda para instalar dependencias
3. ¡Listo para usar!

## 🚀 Uso Rápido

### Clasificación de Razas de Perros
```python
# Ejecutar todo el pipeline en una sola celda
# 1. Dataset sintético generado automáticamente
# 2. Modelo CNN entrenado en 5 épocas
# 3. Predicciones con visualizaciones
# 4. 90%+ accuracy en 2.5 segundos
```

### Clasificación de Noticias
```python
# Pipeline completo de NLP
# 1. Preprocesamiento de texto
# 2. Vectorización TF-IDF
# 3. Entrenamiento con múltiples algoritmos
# 4. Evaluación y métricas completas
```

### Emociones en Audio
```python
# Análisis de audio sin TensorFlow
# 1. Extracción de características (MFCC, spectral)
# 2. Clasificación con Random Forest
# 3. Evaluación en dataset de emociones
```

## 📊 Resultados y Rendimiento

| Proyecto | Framework | Modelo | Accuracy | Tiempo |
|----------|-----------|--------|----------|---------|
| Razas de Perros | PyTorch | CNN | 90%+ | 2.5s |
| Noticias | scikit-learn | RF + TF-IDF | 85%+ | 1-2s |
| Emociones Audio | scikit-learn | Random Forest | 80%+ | 3-5s |

## 🔧 Características Técnicas

### ⚡ Optimizaciones
- **CUDA**: Soporte completo para GPU
- **Mixed Precision**: Entrenamiento acelerado
- **Data Augmentation**: Mejora de generalización
- **Vectorización**: Operaciones optimizadas

### 📱 Compatibilidad
- ✅ **Local**: Windows, Linux, macOS
- ✅ **Google Colab**: Ejecución directa
- ✅ **Kaggle Kernels**: Compatible
- ✅ **GPU/CPU**: Detección automática

### 🛡️ Robustez
- ✅ **Manejo de errores**: Código defensivo
- ✅ **Fallbacks**: Alternativas para dependencias
- ✅ **Logging**: Información detallada de progreso
- ✅ **Reproducibilidad**: Seeds fijos

## 📁 Estructura del Proyecto

```
📦 deep-learning-projects/
├── 📄 README.md                              # Este archivo
├── 📄 requirements.txt                       # Dependencias
├── 📄 LICENSE                               # Licencia MIT
├── 📄 .gitignore                            # Archivos a ignorar
│
├── 🐕 01_clasificacion_razas_perros.ipynb    # CNN para razas de perros
├── 📰 03_clasificacion_noticias.ipynb        # NLP para noticias
├── 🎵 05_clasificacion_emociones_audio.ipynb # Audio emotion recognition
├── 🧠 04_autoencoder_compresion.ipynb       # Autoencoder
├── ✍️ 02_generacion_texto_automatico.ipynb  # Text generation
│
├── 📁 audio_data/                           # Datos de audio
├── 📁 data/                                 # Datasets generales
├── 📁 models/                               # Modelos guardados
└── 📁 utils/                                # Utilidades comunes
```

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👤 Autor

**Cristhian Ismael**
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- LinkedIn: [Tu perfil](https://linkedin.com/in/tu-perfil)

## 🙏 Agradecimientos

- Stanford Dogs Dataset
- PyTorch Community
- scikit-learn Contributors
- Jupyter Project

## 📈 Próximas Mejoras

- [ ] Implementación de modelos Transformer
- [ ] Integración con MLflow para tracking
- [ ] API REST para predicciones
- [ ] Docker containers
- [ ] CI/CD con GitHub Actions
- [ ] Más datasets y benchmarks

---
⭐ ¡Si este proyecto te resulta útil, dale una estrella!
