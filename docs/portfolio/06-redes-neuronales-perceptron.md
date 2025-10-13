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

```python
def perceptron(x1, x2, w1, w2, bias):
    return 1 if (w1*x1 + w2*x2 + bias) >= 0 else 0
```

**Resultados obtenidos:**

- ✅ **AND**: Resuelto exitosamente con pesos w1=0.5, w2=0.5, bias=-1
- ✅ **OR**: Resuelto exitosamente con pesos w1=0.5, w2=0.5, bias=-0.1
- ✅ **NOT**: Resuelto exitosamente con pesos w1=-1, w2=0, bias=0.5
- ❌ **XOR**: Imposible de resolver con perceptrón simple (problema no linealmente separable)

![XOR - Imposible línea recta](07-imagenes/XOR%20-%20imposible%20linea%20recta.png)

### Red Neuronal Multicapa (MLP)

Para resolver el problema XOR, implementamos una red multicapa:

```python
mlp_xor = MLPClassifier(
    hidden_layer_sizes=(6,),
    activation='relu',
    solver='adam',
    random_state=42,
    max_iter=4000
)
```

**Resultado**: ✅ XOR resuelto con 100% de precisión

![MLP para XOR](07-imagenes/MLP%20para%20XOR.png)

![Diferencia entre perceptrón y MLP en el cálculo de XOR](07-imagenes/diferencia%20entre%20perceptron%20y%20MLP%20en%20el%20calculo%20de%20XOR.png)

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

1. **Limitaciones del perceptrón simple**: El perceptrón solo puede resolver problemas linealmente separables, fallando en casos como XOR
2. **Poder de las redes multicapa**: Los MLPs pueden resolver problemas no lineales mediante la combinación de múltiples perceptrones
3. **Importancia de la arquitectura**: El número de neuronas ocultas y capas afecta significativamente el rendimiento
4. **Comparación de frameworks**: TensorFlow/Keras ofrece mayor flexibilidad y control que scikit-learn

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

**¿Por qué AND, OR y NOT funcionaron pero XOR no?** 📏

**R:** Los operadores AND, OR y NOT son **linealmente separables**, es decir, se pueden resolver trazando una línea recta que separe las clases. XOR no es linealmente separable porque no existe una línea recta que pueda separar correctamente los puntos (0,1) y (1,0) de los puntos (0,0) y (1,1). Un perceptrón simple solo puede crear fronteras de decisión lineales.

**¿Cuál es la diferencia clave entre los pesos de AND vs OR?** 🎚️

**R:** La diferencia está en el **umbral (bias)**. AND necesita un umbral más alto (bias=-1) porque requiere que AMBAS entradas sean 1 para activarse. OR tiene un umbral más bajo (bias=-0.1) porque se activa cuando CUALQUIERA de las entradas es 1.

**¿Qué otros problemas del mundo real serían como XOR?** 🚦

**R:** Problemas de **exclusión mutua** como:

- Clasificar si un número es par O impar (pero no ambos)
- Clasificar si un email es spam O legítimo

**¿Por qué sklearn MLP puede resolver XOR pero un perceptrón no?** 🧠

**R:** Un perceptrón simple solo puede crear **una línea de decisión**. Un MLP con capas ocultas puede crear **múltiples líneas de decisión** que se combinan para formar fronteras no lineales. Esto permite resolver problemas como XOR que requieren regiones de decisión más complejas.

**¿Cuál es la principal diferencia entre TensorFlow/Keras y sklearn MLP?** 🔧

**R:** **TensorFlow/Keras** ofrece mucho más control sobre el proceso de entrenamiento (epochs, batch_size, callbacks, optimizadores personalizados), mientras que **sklearn MLP** es más simple pero menos flexible. TensorFlow es mejor para investigación y modelos complejos, sklearn para prototipos rápidos.

**¿Por qué TensorFlow usa epochs y batch_size mientras sklearn MLP no?** ⚙️

**R:** TensorFlow procesa los datos en **mini-batches** (lotes pequeños), lo que permite manejar datasets grandes y actualizar los pesos gradualmente. Sklearn MLP procesa **todo el dataset junto** en cada iteración, lo que es más simple pero menos escalable.

**¿Cuándo usarías sigmoid vs relu como función de activación?** 📊

**R:**
 ReLU es mejor para capas ocultas porque evita el problema del gradiente que desaparece y es computacionalmente eficiente.
 Sigmoid es mejor para la capa de salida en clasificación binaria porque produce valores entre 0 y 1

**¿Qué ventaja tiene PyTorch Lightning sobre TensorFlow puro?** 📝

**R:** PyTorch Lightning reduce significativamente el código necesario para experimentos. Organiza automáticamente el código de entrenamiento, validación y testing, maneja la distribución en múltiples GPUs, y proporciona logging automático.

**¿Por qué PyTorch Lightning separa training_step y test_step?** 🔀

**R:** Durante el entrenamiento se calculan gradientes y se actualizan pesos, mientras que en evaluación solo se hacen predicciones. Lightning separa estos procesos para mayor claridad y para aplicar automáticamente técnicas como dropout solo durante entrenamiento.

**¿Cuál framework elegirías para cada escenario?**
Para prototipo rápido: sklearn MLP por la simplicidad y rapidez, para un modelo en producción TensorFlow/Keras o PyTorch Lightning.

**¿Por qué el error "mat1 and mat2 shapes cannot be multiplied" es común en PyTorch?** 🔍

**R:** Este error ocurre cuando las dimensiones no coinciden entre el dataset y la primera capa del modelo. Por ejemplo, si tu dataset tiene 20 características pero defines la primera capa con 10 neuronas de entrada.

**¿Qué significa el parámetro deterministic=True en PyTorch Lightning Trainer?** 🎲

**R:** Hace que el entrenamiento sea completamente reproducible eliminando la aleatoriedad. Útil para investigación y debugging, pero puede ser más lento. Sin él, cada ejecución puede dar resultados ligeramente diferentes.

**¿Por qué TensorFlow muestra curvas de loss y val_loss durante entrenamiento?** 📈

**R:** Para detectar overfitting visualmente. Si el loss de entrenamiento baja pero el de validación sube, indica que el modelo está memorizando los datos de entrenamiento en lugar de generalizar.

**¿Cuál es la diferencia entre trainer.test() y trainer.predict() en PyTorch Lightning?** 🎯

**R:** trainer.test() calcula métricas de evaluación (accuracy, precision, recall) y trainer.predict() solo genera predicciones sin calcular métricas

**¿Por qué sklearn MLP es más fácil pero menos flexible?** 🛠️

**R:** sklearn abstrae muchos detalles técnicos lo que lo hace fácil de usar, pero se pierde control fino sobre el proceso de entrenamiento, arquitecturas personalizadas y técnicas avanzadas de regularización.

---

> *"Las redes neuronales nos enseñan que la complejidad emerge de la simplicidad, y que múltiples elementos simples pueden resolver problemas que individualmente no podrían abordar"*
