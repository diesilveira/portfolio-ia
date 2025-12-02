# TA9 - CNNs y Transfer Learning: De Redes Convolucionales a Modelos Preentrenados

## Resumen de la Tarea

Esta tarea exploró el mundo de las **Redes Neuronales Convolucionales (CNNs)** y el **Transfer Learning**, dos pilares fundamentales del Deep Learning moderno para visión por computadora. El objetivo fue comprender cómo las CNNs procesan imágenes de manera más efectiva que las redes densas (MLPs), y cómo aprovechar modelos preentrenados para mejorar el rendimiento con menos datos y tiempo de entrenamiento.

### Metodología

1. **Preparación del dataset**: CIFAR-10 con normalización y one-hot encoding
2. **CNN desde cero**: Implementación de una arquitectura convolucional simple
3. **Transfer Learning**: Uso de MobileNetV2 preentrenado en ImageNet
4. **Fine-tuning**: Descongelamiento de capas para ajuste fino
5. **Comparación de arquitecturas**: Evaluación de 9 modelos preentrenados diferentes
6. **Análisis de overfitting**: Comparación de gaps entre train y validation accuracy

## Implementación y Resultados

### Dataset: CIFAR-10

**Características del dataset**: CIFAR-10 contiene 50,000 imágenes de entrenamiento y 10,000 de test, todas con dimensiones de 32×32 píxeles en RGB. El dataset incluye 10 clases balanceadas: airplane, automobile, bird, cat, deer, dog, frog, horse, ship y truck.

### Parte 1: CNN Simple desde Cero

#### Arquitectura

**Características de la arquitectura**: La primera capa convolucional utiliza 32 filtros de 3×3 para detectar patrones básicos como bordes y colores, seguida de MaxPooling que reduce las dimensiones de 32×32 a 16×16. La segunda capa convolucional con 64 filtros de 3×3 detecta patrones más complejos, y otro MaxPooling reduce las dimensiones de 16×16 a 8×8. La capa Flatten convierte la matriz 8×8×64 en un vector de 4,096 elementos, que alimenta una capa Dense de clasificación con 512 neuronas. El modelo resultante tiene 2,122,186 parámetros totales entrenables.

#### Resultados CNN Simple

| Métrica | Valor |
|---------|-------|
| **Training Accuracy** | 86.8% |
| **Validation Accuracy** | 69.4% |
| **Test Accuracy** | 69.37% |
| **Overfitting Gap** | 17.4% |
| **Parámetros** | 2,122,186 |
| **Épocas entrenadas** | 9/10 (EarlyStopping) |

La CNN alcanzó una mejora significativa con 69.37% de accuracy frente al 56.2% del mejor MLP del TA8, aunque presenta overfitting moderado con un gap de 17.4% entre train y validation. El modelo mostró convergencia rápida alcanzando buen rendimiento en solo 9 épocas. Por clase, el mejor desempeño fue en ship (88%), automobile (87%) y frog (84%), mientras que la clase más difícil fue cat (47%).

![Comparación de precisión entre CNN Simple y Transfer Learning](09-imagenes/cnn-vs-transfer-learning.png)

*Gráficas comparativas mostrando la evolución de la precisión en validación durante el entrenamiento (izquierda) y la precisión final de ambos modelos (derecha). La CNN Simple converge más rápido y alcanza mejor rendimiento que Transfer Learning sin fine-tuning.*

### Parte 2: Transfer Learning con MobileNetV2

#### ¿Qué es Transfer Learning?

Transfer Learning utiliza un modelo preentrenado en un dataset grande (como ImageNet con 1.4M imágenes) y lo adapta a nuestro problema específico. Las ventajas principales son entrenamiento más rápido, mejor rendimiento con menos datos, aprovechamiento del conocimiento previo de patrones visuales, y menos parámetros a entrenar.

#### Arquitectura Transfer Learning

**Características**:
- **Base model**: MobileNetV2 preentrenado en ImageNet
- **Capas congeladas**: 2,257,984 parámetros (no se entrenan)
- **Capas entrenables**: 12,810 parámetros (solo clasificador final)
- **Parámetros totales**: 2,270,794

#### Resultados Transfer Learning (Inicial)

| Métrica | Valor |
|---------|-------|
| **Training Accuracy** | 91.8% |
| **Validation Accuracy** | 51.1% |
| **Test Accuracy** | 51.09% |
| **Overfitting Gap** | 40.7% |
| **Parámetros entrenables** | 12,810 |

**Observaciones**: El modelo de Transfer Learning obtuvo un rendimiento significativamente peor que la CNN simple (51.09% vs 69.37%) y presentó un overfitting severo con un gap de 40.7%, el doble que el modelo base. El problema principal es que al mantener el modelo base congelado, este no logra adaptarse correctamente a las características específicas de CIFAR-10.

### Parte 3: Fine-tuning

El fine-tuning consiste en **descongelar las últimas capas** del modelo preentrenado y entrenarlas con un learning rate muy bajo para que se adapten a nuestro dataset específico.

### Parte 4: Comparación de Arquitecturas Preentrenadas

Se evaluaron **9 modelos diferentes** de Keras Applications para identificar cuál funciona mejor en CIFAR-10:

| Ranking | Modelo | Test Acc | Parámetros | Eficiencia* |
|---------|--------|----------|------------|-------------|
| 🥇 | **ResNet50** | 27.02% | 24.1M | 0.011 |
| 🥈 | **ResNet152** | 26.31% | 58.9M | 0.004 |
| 🥉 | **ResNet101** | 26.10% | 43.2M | 0.006 |
| 4 | VGG16 | ~25% | 14.7M | 0.017 |
| 5 | VGG19 | ~24% | 20.0M | 0.012 |
| 6 | EfficientNetB0 | ~23% | 4.0M | 0.058 |
| 7 | EfficientNetB3 | ~22% | 10.7M | 0.021 |
| 8 | MobileNetV2 | ~21% | 2.3M | 0.091 |
| 9 | MobileNetV3Large | ~20% | 2.9M | 0.069 |

*Eficiencia = Test Accuracy / Millones de parámetros

![Comparación de arquitecturas preentrenadas](09-imagenes/model-comparison.png)

*Comparación visual de 9 arquitecturas preentrenadas. Izquierda: Test Accuracy por modelo, donde VGG16 y VGG19 lideran con ~60% de precisión. Derecha: Tamaño del modelo en millones de parámetros, mostrando que ResNet152 es el más grande (60M) mientras que MobileNet son los más eficientes (~2-3M parámetros).*

**Observaciones importantes**: Todos los modelos de transfer learning obtuvieron accuracy muy bajo (20-27%), incluso peor que el baseline MLP (47.4%). Esto se debe a pocas épocas de entrenamiento (solo 5 para comparación rápida), capas base completamente congeladas sin fine-tuning, mismatch de dominios entre ImageNet (224×224) e imágenes de CIFAR-10 (32×32), y configuración subóptima del learning rate y arquitectura del clasificador.

### Análisis Comparativo: CNN vs Transfer Learning

![Comparación de precisión entre CNN Simple y Transfer Learning](09-imagenes/cnn-vs-transfer-learning.png)

*La CNN Simple (azul) muestra una convergencia más estable y alcanza 69.4% de precisión, mientras que Transfer Learning (rojo) sin fine-tuning solo alcanza 51.1%. La diferencia de 18.3% demuestra la importancia de adaptar correctamente los modelos preentrenados al dominio específico.*

## Reflexión y Análisis

### 1. ¿Por qué las CNNs superan a las MLPs en imágenes?

Las CNNs preservan la estructura espacial manteniendo la relación entre píxeles vecinos mientras que las MLPs aplanan la imagen perdiendo esta información, tienen invarianza traslacional donde un filtro detecta el mismo patrón independientemente de su posición (una MLP necesitaría aprender el mismo patrón en cada ubicación), comparten parámetros aplicando los filtros a toda la imagen reduciendo drásticamente el número de parámetros comparado con capas densas, y aprenden automáticamente una jerarquía de características donde la primera capa detecta bordes y colores básicos, la segunda capa texturas y patrones simples, y las capas superiores partes de objetos y objetos completos. Por ejemplo, para detectar un "ojo de gato", una MLP necesita aprender "ojo en posición (10,15)", "ojo en posición (10,16)", etc. requiriendo miles de conexiones, mientras que una CNN aprende un filtro "detector de ojos" que funciona en cualquier posición con solo 9 parámetros (filtro 3×3).

### 2. El Problema del Transfer Learning en CIFAR-10

Sorprendentemente, el transfer learning mostró resultados peores que una CNN simple (51.09% vs 69.37%). Este resultado aparentemente contradictorio se explica por errores en la implementación:

#### Causas del Bajo Rendimiento

**A. Preprocesamiento Incorrecto** ⚠️

El error más grave: se normalizaron todas las imágenes a [0,1] mediante `x/255.0`, pero MobileNetV2 espera imágenes en rango [-1, 1]

Cuando el modelo recibe datos en un rango diferente al que vio durante su entrenamiento en ImageNet, los features extraídos por las capas convolucionales son incorrectos, anulando el beneficio del transfer learning.
**B. Mismatch de Resolución**

- ImageNet: imágenes de **224×224** píxeles
- CIFAR-10: imágenes de **32×32** píxeles (7× más pequeñas por lado)

Los filtros convolucionales entrenados en imágenes grandes no se adaptan bien a imágenes tan pequeñas. Por ejemplo, un filtro 7×7 en una imagen 224×224 captura detalles locales, pero en 32×32 cubre gran parte de la imagen completa.
**C. Configuración Subóptima**

- Aumentar el numero de epocas a 20-30
- Se podria usar alguna tecnica de data augmentation

### 4. Comparación de Modelos Preentrenados

De los 9 modelos evaluados, observamos distintos patrones:

**Modelos grandes (ResNet50, ResNet101, ResNet152)**: Estos modelos tienen mayor capacidad de representación pero son más lentos de entrenar y presentan mayor riesgo de overfitting con pocos datos. El mejor de esta categoría fue ResNet50 con 27.02% de accuracy.

**Modelos eficientes (MobileNet, EfficientNet)**: Son muy rápidos y ligeros, diseñados específicamente para dispositivos móviles, aunque tienen menor capacidad de representación comparados con modelos más grandes. MobileNetV2 destacó con la mejor eficiencia de 0.091 acc/M params.

**Modelos clásicos (VGG16, VGG19)**: Tienen una arquitectura simple y comprensible pero muchos parámetros haciéndolos poco eficientes, y están obsoletos comparados con arquitecturas modernas como ResNet o EfficientNet.

![Comparación de arquitecturas preentrenadas](09-imagenes/model-comparison.png)

*Las gráficas muestran que los modelos VGG obtienen el mejor rendimiento (60%) con tamaño moderado, los MobileNet son muy ligeros pero menos precisos (20-30%), y el ResNet152 siendo el más grande no logra el mejor resultado, demostrando que más parámetros no siempre es mejor.*

### 5. Lecciones Aprendidas

**Sobre CNNs**: Las CNNs son fundamentalmente superiores a las MLPs para visión por computadora, donde incluso una CNN simple supera a MLPs complejas con regularización avanzada. La estructura convolucional captura naturalmente patrones espaciales preservando la información de vecindad entre píxeles, lo que las hace la arquitectura ideal para procesamiento de imágenes.

**Sobre Transfer Learning**: Transfer learning no es una solución mágica y requiere configuración cuidadosa. El mismatch de dominio puede hacer que funcione peor que entrenar desde cero, como observamos en CIFAR-10 donde la diferencia de resolución con ImageNet afectó significativamente los resultados. Cuando funciona bien ahorra tiempo y mejora resultados significativamente, pero el fine-tuning es casi siempre necesario para obtener buenos resultados.

**Sobre el proceso de experimentación**: Es fundamental siempre comparar con un baseline simple como una CNN desde cero para evaluar si el transfer learning realmente aporta valor. Debemos monitorear el overfitting gap y no solo el test accuracy, considerar trade-offs entre accuracy, velocidad y número de parámetros, y probar múltiples arquitecturas antes de tomar una decisión final sobre qué modelo deployar en producción.

## Conclusiones

La superioridad de las CNNs sobre las MLPs quedó claramente demostrada con una CNN simple alcanzando casi 70% de test accuracy, una mejora importante lograda con una arquitectura simple de solo 2 bloques convolucionales. Esta tarea demostró que **la arquitectura importa tanto como los hiperparámetros**; en la TA8 optimizamos exhaustivamente MLPs alcanzando solo 56.2%, mientras que una CNN simple superó ese resultado en la primera iteración.

El transfer learning no funcionó bien "out of the box" debido al mismatch entre ImageNet y CIFAR-10, pero con configuraciones adecuadas (fine-tuning, data augmentation, clasificador más complejo) se podría igualar o superar a la CNN simple. La recomendación es usar CNN desde cero cuando se tiene un dataset suficientemente grande y dominio específico, mientras que transfer learning es preferible con pocos datos, dominio similar a ImageNet o tiempo limitado. Esto ilustra un principio fundamental del deep learning: **usar la arquitectura correcta para el problema correcto es más importante que optimizar una arquitectura incorrecta**.

---

### Recursos adicionales

- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [Transfer Learning Guide - Keras](https://keras.io/guides/transfer_learning/)
- [CIFAR-10 Benchmark](https://paperswithcode.com/sota/image-classification-on-cifar-10)
