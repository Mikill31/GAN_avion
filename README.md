# GAN - Quick, Draw! (Airplane) ✈️

Una Red Neuronal Adversarial Generativa (GAN) desarrollada con PyTorch que aprende a dibujar aviones basándose en los trazos humanos del dataset [Quick, Draw!](https://quickdraw.withgoogle.com/data) de Google.

Este proyecto implementa el concepto de *vibe coding*, estructurando el código de forma declarativa y comentada para explicar la arquitectura y el propósito de cada parámetro interno del Generador y del Discriminador.

## 🚀 Características
* **Red Neuronal Generadora:** Expande un vector de ruido (espacio latente) utilizando capas densas lineales y activaciones `LeakyReLU`, finalizando con `Tanh` para normalizar los píxeles.
* **Red Neuronal Discriminadora:** Actúa como clasificador binario con activaciones `Sigmoid` y capas de `Dropout` (30%) para evitar el sobreajuste (overfitting).
* **Función de Pérdida:** Utiliza `BCELoss` (Binary Cross Entropy) para calcular el nivel de engaño probabilístico entre las redes.
* **Aceleración por Hardware:** Detección automática y soporte para ejecución en GPU (CUDA) o CPU.
* **Visualización:** Generación automática de resultados utilizando `matplotlib` invirtiendo la escala de colores (`gray_r`) para mantener la estética original del juego.

## 🛠️ Requisitos
Necesitas tener Python instalado junto con las siguientes librerías:
```powershell
pip install torch numpy matplotlib

(Nota: Para aceleración por GPU, asegúrate de instalar la versión de PyTorch compatible con CUDA).

