# TA7 - Redes Neuronales: Del Perceptrón Simple al MLP

## Resumen de la Tarea

La **TA7** se enfocó en el estudio de redes neuronales artificiales, comenzando con el perceptrón simple y avanzando hacia redes multicapa (MLP). El objetivo principal fue comprender los fundamentos de las redes neuronales, sus limitaciones y capacidades, implementando desde problemas de lógica booleana hasta clasificación en datasets reales.

### Metodología

1. **Implementación del perceptrón simple**: Desarrollo de la función básica del perceptrón y visualización de su funcionamiento
2. **Problemas de lógica booleana**: Implementación de operadores AND, OR, NOT y XOR
3. **Análisis de limitaciones**: Demostración de por qué el perceptrón simple no puede resolver XOR
4. **Redes multicapa (MLP)**: Implementación usando scikit-learn para resolver problemas no linealmente separables
5. **Comparación con TensorFlow/Keras**: Implementación de una red neuronal profesional
6. **Evaluación en dataset real**: Aplicación a un problema de clasificación más complejo

### Conceptos Clave

- **Perceptrón**: Unidad básica de procesamiento que implementa una función de activación lineal
- **Función de activación**: Determina la salida de una neurona basada en sus entradas
- **Separabilidad lineal**: Capacidad de separar clases con una línea recta (limitación del perceptrón simple)
- **MLP**: Red neuronal multicapa capaz de resolver problemas no linealmente separables

## Implementación y Resultados

### Perceptrón Simple

**Resultados obtenidos:**

- ✅ **AND**: Resuelto exitosamente con pesos w1=0.5, w2=0.5, bias=-1
- ✅ **OR**: Resuelto exitosamente con pesos w1=0.5, w2=0.5, bias=-0.1
- ✅ **NOT**: Resuelto exitosamente con pesos w1=-1, w2=0, bias=0.5
- ❌ **XOR**: Imposible de resolver con perceptrón simple

![XOR - Imposible línea recta](07-imagenes/XOR%20-%20imposible%20linea%20recta.png)

*Cuatro intentos de separar XOR con una línea recta, todos fallan. Los puntos azules (0,1) y (1,0) deben estar en una clase, mientras que los rojos (0,0) y (1,1) en otra. No importa cómo se trace la línea, siempre clasifica incorrectamente al menos un punto. Esto demuestra por qué un perceptrón simple no puede resolver XOR.*

### Red Neuronal Multicapa (MLP)

- ✅ XOR resuelto.

![MLP para XOR](07-imagenes/MLP%20para%20XOR.png)

*Arquitectura del MLP para resolver XOR: 2 neuronas de entrada (x1, x2), 6 neuronas en la capa oculta, y 1 neurona de salida. Cada línea representa una conexión con peso ajustable. Esta arquitectura permite crear la superficie necesaria para separar correctamente los puntos de XOR.*

![Diferencia entre perceptrón y MLP en el cálculo de XOR](07-imagenes/diferencia%20entre%20perceptron%20y%20MLP%20en%20el%20calculo%20de%20XOR.png)

*Comparación visual: Izquierda - El perceptrón solo puede trazar una línea recta que no logra separar XOR correctamente. Derecha - El MLP crea una superficie que separa perfectamente las clases, demostrando por qué necesitamos capas ocultas para problemas no lineales.*

### Aplicación en Dataset Real

Implementamos un MLP para un dataset de clasificación más complejo:

| Modelo | Arquitectura | Training Accuracy | Test Accuracy |
|--------|-------------|-------------------|---------------|
| Scikit-learn MLP | 20 → (100, 50) → 2 | 100.0% | 93.0% |
| TensorFlow/Keras | 20 → (64, 128) → 1 | 99.7% | 96.0% |

## Visualizaciones Principales

El análisis incluyó múltiples visualizaciones:

1. **Separación lineal**: Visualización de cómo el perceptrón separa clases con una línea recta
2. **Problema XOR**: Demostración visual de por qué XOR no es linealmente separable
3. **Superficies de decisión**: Comparación entre perceptrón simple y MLP
4. **Matrices de confusión**: Evaluación del rendimiento en datasets reales
5. **Curvas de aprendizaje**: Análisis del entrenamiento con TensorFlow

## Reflexión

### Hallazgos Principales

Esta tarea demostró las limitaciones del perceptrón simple que solo puede resolver problemas linealmente separables fallando en casos como XOR, mientras que los MLPs pueden resolver problemas no lineales mediante la combinación de múltiples perceptrones en capas ocultas. El número de neuronas ocultas y capas afecta significativamente el rendimiento del modelo, y frameworks como TensorFlow/Keras ofrecen mayor flexibilidad y control que scikit-learn para diseñar arquitecturas personalizadas.

### Desafíos Encontrados

- **Selección de hiperparámetros**: Encontrar la arquitectura óptima requiere experimentación
- **Overfitting**: Los modelos complejos pueden memorizar los datos de entrenamiento
- **Interpretabilidad**: Las redes neuronales son "cajas negras" difíciles de interpretar
- **Tiempo de entrenamiento**: Los modelos más complejos requieren más tiempo computacional

### Comparaciones y Mejoras

- **Perceptrón vs MLP**: El MLP supera las limitaciones del perceptrón simple pero es más complejo
- **Scikit-learn vs TensorFlow**: TensorFlow ofrece mayor control pero requiere más código
- **Arquitecturas diferentes**: Más capas y neuronas mejoran la capacidad pero aumentan el riesgo de overfitting

### Preguntas de Reflexión y Respuestas

**¿Por qué AND, OR y NOT funcionaron pero XOR no?**

Los operadores AND, OR y NOT son linealmente separables, es decir, se pueden resolver trazando una línea recta que separe las clases. XOR no es linealmente separable porque no existe una línea recta que pueda separar correctamente los puntos (0,1) y (1,0) de los puntos (0,0) y (1,1), y un perceptrón simple solo puede crear fronteras de decisión lineales.

**¿Cuál es la diferencia clave entre los pesos de AND vs OR?**

La diferencia está en el umbral (bias). AND necesita un umbral más alto (bias=-1) porque requiere que ambas entradas sean 1 para activarse, mientras que OR tiene un umbral más bajo (bias=-0.1) porque se activa cuando cualquiera de las entradas es 1.

**¿Qué otros problemas del mundo real serían como XOR?**

Problemas de exclusión mutua como clasificar si un número es par o impar (pero no ambos), o clasificar si un email es spam o legítimo, donde las categorías son mutuamente excluyentes y requieren fronteras de decisión no lineales.

**¿Por qué sklearn MLP puede resolver XOR pero un perceptrón no?**

Un perceptrón simple solo puede crear una línea de decisión, mientras que un MLP con capas ocultas puede crear múltiples líneas de decisión que se combinan para formar fronteras no lineales, lo que permite resolver problemas como XOR que requieren regiones de decisión más complejas.

**¿Cuál es la principal diferencia entre TensorFlow/Keras y sklearn MLP?**

TensorFlow/Keras ofrece mucho más control sobre el proceso de entrenamiento (epochs, batch_size, callbacks, optimizadores personalizados), mientras que sklearn MLP es más simple pero menos flexible. TensorFlow es mejor para investigación y modelos complejos, sklearn para prototipos rápidos.

**¿Por qué TensorFlow usa epochs y batch_size mientras sklearn MLP no?**

TensorFlow procesa los datos en mini-batches (lotes pequeños), lo que permite manejar datasets grandes y actualizar los pesos gradualmente. Sklearn MLP procesa todo el dataset junto en cada iteración, lo que es más simple pero menos escalable para grandes volúmenes de datos.

**¿Cuándo usarías sigmoid vs relu como función de activación?**

ReLU es mejor para capas ocultas porque evita el problema del gradiente que desaparece y es computacionalmente eficiente, mientras que sigmoid es mejor para la capa de salida en clasificación binaria porque produce valores entre 0 y 1 que se pueden interpretar como probabilidades.

**¿Qué ventaja tiene PyTorch Lightning sobre TensorFlow puro?**

PyTorch Lightning reduce significativamente el código necesario para experimentos, organiza automáticamente el código de entrenamiento, validación y testing, maneja la distribución en múltiples GPUs, y proporciona logging automático, haciendo el desarrollo más eficiente y organizado.

**¿Por qué PyTorch Lightning separa training_step y test_step?**

Durante el entrenamiento se calculan gradientes y se actualizan pesos, mientras que en evaluación solo se hacen predicciones. Lightning separa estos procesos para mayor claridad y para aplicar automáticamente técnicas como dropout solo durante entrenamiento, no durante evaluación.

**¿Cuál framework elegirías para cada escenario?**

Para prototipo rápido elegiría sklearn MLP por la simplicidad y rapidez de implementación, mientras que para un modelo en producción optaría por TensorFlow/Keras o PyTorch Lightning que ofrecen mayor control, escalabilidad y herramientas avanzadas de monitoreo.

**¿Por qué el error "mat1 and mat2 shapes cannot be multiplied" es común en PyTorch?**

Este error ocurre cuando las dimensiones no coinciden entre el dataset y la primera capa del modelo. Por ejemplo, si tu dataset tiene 20 características pero defines la primera capa con 10 neuronas de entrada, PyTorch no puede realizar la multiplicación matricial necesaria.

**¿Qué significa el parámetro deterministic=True en PyTorch Lightning Trainer?**

Este parámetro hace que el entrenamiento sea completamente reproducible eliminando la aleatoriedad. Es útil para investigación y debugging, pero puede ser más lento. Sin él, cada ejecución puede dar resultados ligeramente diferentes debido a la inicialización aleatoria y el orden de procesamiento.

**¿Por qué TensorFlow muestra curvas de loss y val_loss durante entrenamiento?**

Para detectar overfitting visualmente. Si el loss de entrenamiento baja pero el de validación sube, indica que el modelo está memorizando los datos de entrenamiento en lugar de generalizar, permitiendo tomar decisiones como aplicar regularización o detener el entrenamiento.

**¿Cuál es la diferencia entre trainer.test() y trainer.predict() en PyTorch Lightning?**

trainer.test() calcula métricas de evaluación como accuracy, precision y recall comparando predicciones con labels reales, mientras que trainer.predict() solo genera predicciones sin calcular métricas, útil cuando no se tienen labels disponibles.

**¿Por qué sklearn MLP es más fácil pero menos flexible?**

sklearn abstrae muchos detalles técnicos lo que lo hace fácil de usar con pocas líneas de código, pero se pierde control fino sobre el proceso de entrenamiento, arquitecturas personalizadas, técnicas avanzadas de regularización y optimización que frameworks como TensorFlow o PyTorch sí permiten.

