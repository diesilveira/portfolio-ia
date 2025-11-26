# TA3 - Regresión Lineal y Logística

## Resumen de la Tarea

Esta tarea consistió en implementar y comparar dos tipos  de modelos de machine learning: **Regresión Lineal** para problemas de predicción continua y **Regresión Logística** para problemas de clasificación binaria. El objetivo principal fue comprender las diferencias entre ambos tipos de modelos, sus métricas de evaluación específicas y cuándo aplicar cada uno.

### Metodología

1. **Regresión Lineal - Dataset Boston Housing**:
   - Predicción de precios de viviendas (variable continua)
   - Preprocesamiento con imputación de valores faltantes
   - Evaluación con métricas de regresión (MAE, RMSE)
   - Análisis de importancia de características

2. **Regresión Logística - Dataset Breast Cancer**:
   - Clasificación binaria: tumor benigno vs maligno
   - División estratificada para mantener proporción de clases
   - Evaluación con métricas de clasificación (Accuracy, Precision, Recall, F1)
   - Análisis de matriz de confusión

### Datasets Utilizados

**🏠 Boston Housing (Regresión)**
- **Tamaño**: 506 registros con 13 características
- **Target**: Precios de viviendas ($5.0k - $50.0k)
- **Problema**: Predicción de valores continuos

**🏥 Breast Cancer (Clasificación)**
- **Tamaño**: 569 pacientes con 30 características
- **Target**: Diagnóstico binario (212 malignos, 357 benignos)
- **Problema**: Clasificación binaria

## Resultados de los Modelos

### Regresión Lineal - Boston Housing

**Métricas de Rendimiento:**
- **MAE**: $3.02k (error promedio absoluto)
- **RMSE**: $4.76k (error cuadrático medio)

**Variables más importantes:**
1. **NOX** (-16.75): Nivel de óxidos nitrosos (negativo = a mayor contaminación, menor precio)
2. **RM** (+4.11): Número de habitaciones (positivo = más habitaciones, mayor precio)
3. **CHAS** (+3.05): Proximidad al río Charles (positivo = cerca del río, mayor precio)

### Regresión Logística - Breast Cancer

**Métricas de Rendimiento:**
- **Accuracy**: 95.6%
- **Precision**: 94.6% (de los predichos como benignos, 94.6% lo son realmente)
- **Recall**: 98.6% (de todos los benignos reales, detectamos 98.6%)
- **F1-Score**: 0.966

**Matriz de Confusión:**
```
           Predicho
Actual   Maligno  Benigno
Maligno     39      4
Benigno      1     70
```

## Parte 1: Métricas de Regresión

**MAE (Mean Absolute Error)**: Promedio de los errores **absolutos** sin importar si son positivos o negativos.

**MSE (Mean Squared Error)**: Promedio de los errores **al cuadrado**, penaliza más los errores grandes.

**RMSE**: Raíz cuadrada del MSE, vuelve a las **unidades** originales del problema.

**R²**: Indica qué porcentaje de la **varianza** es explicada por el modelo (0-1, donde 1 es perfecto).

**MAPE**: Error porcentual promedio, útil para comparar modelos con diferentes **escalas**.

### Interpretación en el TA3:
- **MAE = $3.02k**: En promedio, nuestras predicciones se alejan $3,020 del precio real
- **RMSE = $4.76k**: La raíz del error cuadrático medio es $4,760, penalizando más los errores grandes
- El RMSE > MAE indica que hay algunos errores grandes que afectan más la métrica cuadrática

## Parte 2: Métricas de Clasificación

**Accuracy**: Porcentaje de predicciones **correctas** sobre el total.

**Precision**: De todas las predicciones **positivas**, ¿cuántas fueron realmente correctas?

**Recall (Sensibilidad)**: De todos los casos **positivos** reales, ¿cuántos detectamos?

**F1-Score**: Promedio **armónico** entre precision y recall.

**Matriz de Confusión**: Tabla que muestra **predicciones** vs **valores reales**.

### Interpretación en el TA3:
- **Accuracy = 95.6%**: De 114 casos, acertamos en 109
- **Precision = 94.6%**: De los 74 casos que predijimos como benignos, 70 realmente lo eran
- **Recall = 98.6%**: De los 71 casos benignos reales, detectamos 70
- **F1-Score = 0.966**: Excelente balance entre precision y recall

## Parte 3: Selección de Modelos

### ¿Cuál modelo usarías para predecir el salario de un empleado?

Usaría **Regresión Lineal** porque el salario es una **variable continua** (puede tomar cualquier valor numérico dentro de un rango). Los salarios no son categorías discretas sino números que pueden variar gradualmente (ej: $45,000, $45,500, $46,000, etc.). La regresión lineal es ideal para predecir valores numéricos continuos basándose en características como experiencia, educación, ubicación, etc.

### ¿Cuál modelo usarías para predecir si un email es spam?

Usaría **Regresión Logística** porque la clasificación de emails es un problema de **clasificación binaria**: solo hay dos opciones posibles (spam o no spam). No necesitamos predecir un valor numérico continuo, sino asignar cada email a una de dos categorías mutuamente excluyentes. La regresión logística es perfecta para este tipo de decisiones binarias ya que puede calcular la probabilidad de que un email sea spam y clasificarlo según un umbral.

### ¿Por qué es importante separar datos de entrenamiento y prueba?

Para **evitar el sobreajuste (overfitting)** y obtener una evaluación honesta del modelo. Si evaluáramos el modelo con los mismos datos que usamos para entrenarlo, obtendríamos una medida artificialmente optimista del rendimiento - como si un estudiante se evaluara con las mismas preguntas que estudió.

**Razones específicas:**
- **Validación independiente**: Los datos de prueba actúan como un "examen final" que el modelo nunca ha visto
- **Detección de overfitting**: Si el modelo funciona bien en entrenamiento pero mal en prueba, está memorizando en lugar de generalizar
- **Estimación realista**: El rendimiento en datos de prueba nos dice cómo se comportará el modelo con datos nuevos en producción
- **Selección de modelos**: Podemos comparar diferentes modelos usando la misma métrica en el mismo conjunto de prueba

En el TA3, usamos 80% para entrenamiento y 20% para prueba, asegurándonos de que nuestras métricas (MAE, RMSE, Accuracy, etc.) reflejen el verdadero poder predictivo de los modelos.

## Conclusiones

1. **Diferencias fundamentales**: Regresión para valores continuos vs Clasificación para categorías discretas
2. **Métricas específicas**: Cada tipo de problema requiere métricas de evaluación apropiadas
3. **Importancia de la validación**: La separación train/test es crucial para evaluaciones honestas
4. **Aplicación práctica**: Ambos modelos lograron excelente rendimiento en sus respectivos dominios

Los resultados confirman que elegir el modelo correcto según la naturaleza del problema es fundamental para el éxito en machine learning.
