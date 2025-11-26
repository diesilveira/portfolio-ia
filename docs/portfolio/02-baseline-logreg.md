# TA2 - Modelo Baseline y Regresión Logística

## Resumen de la Tarea

El **TA2** consistió en implementar un modelo baseline y una regresión logística para predecir la supervivencia en el Titanic. El objetivo principal fue comparar el rendimiento de un modelo simple (baseline) con un modelo de machine learning más sofisticado, aplicando técnicas de feature engineering y preprocesamiento de datos.

### Metodología

1. **Preprocesamiento de datos**: 
   - Imputación de valores faltantes en `Age`, `Fare` y `Embarked`
   - Manejo estratégico de missing values según el tipo de variable

2. **Feature Engineering**:
   - Creación de `FamilySize` (SibSp + Parch + 1)
   - Variable `IsAlone` para identificar pasajeros solitarios
   - Extracción de títulos del nombre (`Title`)
   - Agrupación de títulos raros bajo categoría "Rare"

3. **Modelado**:
   - **Baseline**: DummyClassifier que predice siempre la clase más común
   - **Regresión Logística**: Modelo supervisado con regularización
   - División train/test (80/20) con estratificación

4. **Evaluación**:
   - Métricas de accuracy, precision, recall y F1-score
   - Matriz de confusión para análisis de errores

### Dataset Procesado

- **Features utilizadas**: `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `FamilySize`, `IsAlone`, `Title`, `SibSp`, `Parch`
- **Encoding**: Variables categóricas convertidas con `pd.get_dummies()`
- **Tamaño final**: 891 registros con features procesadas

## Resultados del Modelo

### Métricas de Rendimiento

- **Baseline Accuracy**: 61.45%
- **Regresión Logística Accuracy**: 81.56%

### Matriz de Confusión (Regresión Logística)
```
           Predicho
Actual    No (0)  Sí (1)
No (0)      98     12
Sí (1)      21     48
```

### Reporte de Clasificación
```
              precision    recall  f1-score   support
No (0)           0.82      0.89      0.86       110
Sí (1)           0.80      0.70      0.74        69
accuracy                            0.82       179
```

## Análisis de Resultados

### Matriz de confusión: ¿En qué casos se equivoca más el modelo?

El modelo se equivoca más cuando **predice que una persona NO sobrevivió, pero sí lo hizo** (21 casos vs 12 casos). Específicamente:

- **Falsos Negativos (FN)**: 21 casos - Predice "No sobrevive" pero sí sobrevivió
- **Falsos Positivos (FP)**: 12 casos - Predice "Sobrevive" pero no sobrevivió

Esto indica que el modelo es más **conservador** y tiende a subestimar las probabilidades de supervivencia.

### Clases atendidas: ¿El modelo acierta más con los que sobrevivieron o con los que no sobrevivieron?

El modelo acierta **más con los que NO sobrevivieron**:

- **Clase "No sobrevivió" (0)**: Recall = 89% (98 de 110 casos correctos)
- **Clase "Sobrevivió" (1)**: Recall = 70% (48 de 69 casos correctos)

El modelo tiene mejor capacidad para identificar a las personas que no sobrevivieron, posiblemente porque esta clase está mejor representada en los datos de entrenamiento.

### Comparación con baseline: ¿La Regresión Logística obtiene más aciertos?

Si, la Regresión Logística supera ampliamente al baseline:

- **Baseline (siempre clase más común)**: 61.45% accuracy
- **Regresión Logística**: 81.56% accuracy
- **Mejora**: +20.11 puntos porcentuales

Esto demuestra que las features engineered y el modelo de ML aportan valor predictivo real.

### Errores más importantes: ¿Cuál de los dos tipos de error es más grave?

Para el problema del Titanic, los **Falsos Negativos (FN) son más graves**:

- **FN (21 casos)**: Predecir que alguien NO sobrevivirá cuando SÍ lo hará
- **FP (12 casos)**: Predecir que alguien sobrevivirá cuando NO lo hará

En un contexto de emergencia, es más crítico **no identificar a alguien que podría salvarse** (FN) que generar una falsa esperanza (FP). Los FN representan oportunidades perdidas de rescate.

### Observaciones generales: Patrones interesantes sobre supervivencia

1. **Desequilibrio de clases**: El modelo maneja mejor la clase mayoritaria (No sobrevivió)

2. **Feature Engineering efectivo**: Las nuevas variables (`FamilySize`, `IsAlone`, `Title`) contribuyeron significativamente al rendimiento

3. **Precision vs Recall trade-off**: 
   - Alta precision para "No sobrevivió" (82%)
   - Menor recall para "Sobrevivió" (70%)

4. **Patrón socioeconómico**: El modelo captura bien las diferencias de clase y género identificadas en el TA1

## Conclusiones

1. **El feature engineering es muy importante**: Las nuevas variables mejoraron significativamente el rendimiento
2. **La regresión logística supera al baseline**: +20% de accuracy
3. **El modelo tiene sesgo conservador**: Tiende a predecir más muertes que supervivencias
