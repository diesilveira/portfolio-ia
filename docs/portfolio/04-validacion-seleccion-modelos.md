# TA4 - Validación y Selección de Modelos

## Resumen de la Tarea

Esta tarea consistió en implementar técnicas avanzadas de validación cruzada y selección de modelos para evaluar la estabilidad y rendimiento de diferentes algoritmos de machine learning. El objetivo principal fue comprender la importancia de la validación rigurosa, comparar métodos de validación cruzada, y seleccionar el mejor modelo basándose en criterios de rendimiento y estabilidad.

### Metodología

1. **Dataset utilizado**: Student Dropout and Academic Success (UCI ML Repository)
   - 4,424 estudiantes con 36 características
   - Problema multiclase: Dropout (32.1%), Enrolled (17.9%), Graduate (49.9%)

2. **Técnicas de validación cruzada**:
   - **KFold**: División simple en 5 partes
   - **StratifiedKFold**: Mantiene proporción de clases en cada fold

3. **Comparación de modelos**:
   - Logistic Regression con StandardScaler
   - Ridge Classifier con regularización L2
   - Random Forest (ensemble method)

4. **Métricas de evaluación**:
   - F1-score weighted para manejar clases desbalanceadas
   - Análisis de estabilidad mediante desviación estándar
   - Visualización comparativa de distribuciones

### Dataset: Student Dropout and Academic Success

- **Estudiantes**: 4,424 registros
- **Características**: 36 variables (edad al inscribirse, calificaciones previas, estado civil, etc.)
- **Clases**: Dropout (0), Enrolled (1), Graduate (2)
- **Distribución**: Desbalanceada con predominio de graduados (49.9%)

## Resultados de Validación Cruzada

### Comparación KFold vs StratifiedKFold

**KFold (5 splits):**
- Scores individuales: [0.742, 0.745, 0.755, 0.762, 0.765]
- **Resultado**: 75.36% ± 0.89%

**StratifiedKFold (5 splits):**
- Scores individuales: [0.755, 0.746, 0.754, 0.749, 0.751]
- **Resultado**: 75.09% ± 0.32%

**Conclusión**: StratifiedKFold es **MÁS ESTABLE** (menor variabilidad: 0.32% vs 0.89%)

### Competencia de Modelos

**Resultados con 5-Fold Cross-Validation (F1-weighted):**

1. **🏆 Random Forest**: 74.94% ± 0.41% (GANADOR)
2. **Logistic Regression**: 74.42% ± 0.71%
3. **Ridge Classifier**: 70.96% ± 0.47%

**Análisis de estabilidad**: Todos los modelos son **MUY ESTABLES** (std < 0.02)

![Gráficos de Validación Cruzada y Comparación de Modelos](Practica 4 graficas.png)

Los gráficos muestran la distribución de scores y la comparación visual entre los diferentes métodos de validación cruzada y modelos evaluados.

## Parte 1: Definiciones de Cross-Validation

### Completa las definiciones:

**Cross-Validation**: Técnica que divide los datos en **múltiples** partes para entrenar y evaluar múltiples veces.

**Accuracy promedio**: La **estimación** de rendimiento esperado en datos nuevos.

**Desviación estándar**: Indica qué tan **estable** es el modelo entre diferentes divisiones de datos.

**StratifiedKFold**: Mantiene la **proporción** de clases en cada fold, especialmente importante en datasets desbalanceados.

### Interpretación:
- **Cross-validation 5-fold**: Dividimos en 5 partes, entrenamos 5 veces diferentes
- **75.09% ± 0.32%**: Rendimiento esperado con alta confianza
- **Baja desviación (0.32%)**: Modelo muy consistente entre diferentes divisiones

## Parte 2: ¿Cuándo usar cada método?

### Completa la guía de decisión:

GridSearchCV cuando se tiene pocos hiperparámetros y suficiente tiempo de cómputo.

RandomizedSearchCV cuando tienes muchos hiperparámetros o tiempo** limitado.

Pipeline + SearchCV siempre previene data leakage automáticamente.

cross_val_score en el resultado final valida que la optimización no causó** overfitting.

### Aplicación práctica:
- **GridSearchCV**: Ideal para optimizar 2-3 hiperparámetros con búsqueda exhaustiva
- **RandomizedSearchCV**: Mejor para Random Forest con muchos parámetros (n_estimators, max_depth, etc.)
- **Pipeline**: En el TA4 previno que StandardScaler viera datos de test durante CV
- **cross_val_score**: Confirmó que Random Forest no se sobreajustó durante la selección

## Parte 3: ¿Por qué es importante la explicabilidad?

### Completa las razones:

**Confianza**: Los educadores necesitan **entender** por qué el modelo predice abandono.

**Intervenciones**: Knowing las características importantes permite crear **estrategias** específicas.

**Bias detection**: La explicabilidad ayuda a detectar **sesgos** en el modelo.

**Regulaciones**: Muchos contextos requieren modelos **interpretables** por ley.

**Mejora continua**: Entender el modelo ayuda a **mejorar** futuras versiones.

## Parte 4: Preguntas de Reflexión

### ¿Qué es data leakage y por qué es peligroso?

**Respuesta**: Data leakage ocurre cuando información que no debería estar disponible durante el entrenamiento "se filtra" al modelo, creando una ventaja artificial que no existirá en producción.

### ¿Cuándo usar KFold vs StratifiedKFold?

**KFold cuando:**
- Problemas de regresión (no hay clases)
- Datasets muy grandes donde la proporción se mantiene naturalmente

**StratifiedKFold cuando:**
- **Clases desbalanceadas**
- **Clases minoritarias** que podrían desaparecer en algunos folds
- **Siempre en clasificación** como práctica defensiva

En esta tarea: StratifiedKFold fue esencial porque con solo 17.9% de estudiantes "Enrolled", KFold podría crear folds sin esta clase.

### ¿Cómo interpretar "75.09% ± 0.32%" en cross-validation?

**Interpretación completa:**
- **75.09%**: Rendimiento promedio esperado en datos nuevos
- **± 0.32%**: Intervalo de confianza - el modelo es muy estable
- **Rango**: Entre 74.77% y 75.41% en el 68% de los casos
- **Estabilidad**: Desviación muy baja indica modelo robusto

**Comparación**: 75.09% ± 0.32% es mucho mejor que 75.36% ± 0.89% porque la menor variabilidad indica mayor confiabilidad.

### ¿Por qué Random Forest no necesita StandardScaler?

**Respuesta**: Random Forest está basado en **árboles de decisión**, que toman decisiones mediante **comparaciones ordinales** (mayor/menor) en lugar de **distancias euclidianas**.

Por eso Random Forest no incluye StandardScaler en su Pipeline, mientras que Logistic Regression y Ridge sí lo necesitan.

### En diagnóstico médico, ¿prefieres 98% accuracy pero inestable, o 95% accuracy pero muy estable?

**Respuesta**: **95% accuracy pero muy estable**

**Justificación:**
- **Confiabilidad**: Un modelo que varía mucho genera desconfianza

- **Decisiones consistentes**: 

- **Implementación**: Un modelo estable es más fácil de mantener y actualizar

- **Riesgo**: 3% menos de accuracy es preferible a predicciones erráticas


## Conclusiones

El TA4 demostró exitosamente:

1. **StratifiedKFold es superior** para datasets desbalanceados (0.32% vs 0.89% de variabilidad)
2. **Random Forest ganó** la competencia con 74.94% ± 0.41% de F1-score
3. **Todos los modelos fueron estables** (std < 0.02), indicando validación robusta
4. **Los Pipelines previenen data leakage** automáticamente en validación cruzada
5. **La estabilidad es tan importante como el rendimiento** en aplicaciones reales

Los resultados confirman que una validación rigurosa es fundamental para seleccionar modelos confiables que funcionen bien en producción, especialmente en contextos sensibles como la educación donde las decisiones afectan el futuro de los estudiantes.
