"""
🧪 Guía de Experimentación Sistemática - CIFAR-10 MLP
=====================================================

Este script implementa todos los experimentos sugeridos en las guías de experimentación.

Estructura:
1. PARTE 1: Arquitecturas (profundidad, ancho, activaciones, BatchNorm, Dropout, L2, inicializadores)
2. PARTE 2: Optimizadores (Adam, SGD, RMSprop, AdamW con diferentes hiperparámetros)
3. PARTE 3: Callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler)

Para ejecutar en notebook: copiar cada sección en una celda separada
"""

import os, math, json, time, random, datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, initializers, callbacks

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPU disponibles:", tf.config.list_physical_devices('GPU'))

# Carpetas para logs y checkpoints
ROOT_LOGDIR = "tb_logs_experiments"
CHECKPOINT_DIR = "model_checkpoints"
os.makedirs(ROOT_LOGDIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================================
# CARGA DE DATOS
# ============================================================================

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.flatten(); y_test = y_test.flatten()

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# Normalización a [-1, 1]
x_train = (x_train.astype("float32")/255.0 - 0.5) * 2.0
x_test  = (x_test.astype("float32")/255.0 - 0.5) * 2.0

# Split de validación (10% del train)
VAL_RATIO = 0.1
n_val = int(len(x_train)*VAL_RATIO)
x_val, y_val = x_train[:n_val], y_train[:n_val]
x_train, y_train = x_train[n_val:], y_train[n_val:]

# APLANAR imágenes 32x32x3 -> vectores 3072
x_train = x_train.reshape(len(x_train), -1)
x_val   = x_val.reshape(len(x_val), -1)
x_test  = x_test.reshape(len(x_test), -1)

print("Train:", x_train.shape, "Val:", x_val.shape, "Test:", x_test.shape)

# ============================================================================
# UTILIDADES PARA EXPERIMENTACIÓN
# ============================================================================

results_db = []

def run_experiment(model, experiment_name, epochs=20, batch_size=64, 
                   use_callbacks=None, verbose=1):
    """Ejecuta un experimento completo y guarda los resultados."""
    
    print(f"\n{'='*60}")
    print(f"🧪 Experimento: {experiment_name}")
    print(f"{'='*60}")
    
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{experiment_name}_{timestamp}"
    
    # Configurar callbacks
    run_dir = os.path.join(ROOT_LOGDIR, run_name)
    all_callbacks = [keras.callbacks.TensorBoard(log_dir=run_dir, histogram_freq=1)]
    
    if use_callbacks:
        all_callbacks.extend(use_callbacks)
    
    # Entrenar
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=verbose,
        callbacks=all_callbacks
    )
    training_time = time.time() - start_time
    
    # Evaluar
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Mejor época
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = np.max(history.history['val_accuracy'])
    
    # Guardar resultados
    result = {
        'experiment_name': experiment_name,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'total_params': model.count_params(),
        'training_time': training_time,
        'epochs': epochs,
        'batch_size': batch_size,
        'history': history.history
    }
    
    results_db.append(result)
    
    # Imprimir resultados
    print(f"\n📊 Resultados {experiment_name}:")
    print(f"  Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"  Mejor época: {best_epoch}/{epochs} (val_acc={best_val_acc:.4f})")
    print(f"  Parámetros: {model.count_params():,}")
    print(f"  Tiempo: {training_time:.1f}s")
    
    return result

def compare_experiments(results_list):
    """Compara múltiples experimentos."""
    df = pd.DataFrame([{
        'Experimento': r['experiment_name'],
        'Train Acc': r['train_acc'],
        'Val Acc': r['val_acc'],
        'Test Acc': r['test_acc'],
        'Val Loss': r['val_loss'],
        'Parámetros': r['total_params'],
        'Tiempo (s)': r['training_time'],
        'Mejor Época': r['best_epoch']
    } for r in results_list])
    
    return df.sort_values('Test Acc', ascending=False)

# ============================================================================
# PARTE 1: EXPERIMENTOS DE ARQUITECTURA
# ============================================================================

print("\n" + "="*80)
print("🏗️ PARTE 1: EXPERIMENTACIÓN CON ARQUITECTURAS")
print("="*80)

# --- Experimento 1.1: Baseline ---
print("\n### Experimento 1.1: Modelo Baseline (2 capas × 32 neuronas)")
model_baseline = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])
model_baseline.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_baseline = run_experiment(model_baseline, "01_baseline", epochs=20, batch_size=64)

# --- Experimento 1.2: Mayor profundidad ---
print("\n### Experimento 1.2: Mayor profundidad (4 capas × 64 neuronas)")
model_deep = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])
model_deep.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_deep = run_experiment(model_deep, "02_deep_4x64", epochs=20, batch_size=64)

# --- Experimento 1.3: Mayor ancho ---
print("\n### Experimento 1.3: Mayor ancho (3 capas × 128 neuronas)")
model_wide = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])
model_wide.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_wide = run_experiment(model_wide, "03_wide_3x128", epochs=20, batch_size=64)

# --- Experimento 1.4: Arquitectura piramidal ---
print("\n### Experimento 1.4: Arquitectura piramidal (256→128→64)")
model_pyramid = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])
model_pyramid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_pyramid = run_experiment(model_pyramid, "04_pyramid", epochs=20, batch_size=64)

# --- Experimento 1.5: Activación GELU ---
print("\n### Experimento 1.5: Activación GELU")
model_gelu = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(128, activation='gelu'),
    layers.Dense(128, activation='gelu'),
    layers.Dense(128, activation='gelu'),
    layers.Dense(len(class_names), activation='softmax')
])
model_gelu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_gelu = run_experiment(model_gelu, "05_gelu", epochs=20, batch_size=64)

# --- Experimento 1.6: Activación Tanh ---
print("\n### Experimento 1.6: Activación Tanh")
model_tanh = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(128, activation='tanh'),
    layers.Dense(128, activation='tanh'),
    layers.Dense(128, activation='tanh'),
    layers.Dense(len(class_names), activation='softmax')
])
model_tanh.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_tanh = run_experiment(model_tanh, "06_tanh", epochs=20, batch_size=64)

# --- Experimento 1.7: BatchNormalization ---
print("\n### Experimento 1.7: Con BatchNormalization")
model_bn = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(128), layers.BatchNormalization(), layers.Activation('relu'),
    layers.Dense(128), layers.BatchNormalization(), layers.Activation('relu'),
    layers.Dense(128), layers.BatchNormalization(), layers.Activation('relu'),
    layers.Dense(len(class_names), activation='softmax')
])
model_bn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_bn = run_experiment(model_bn, "07_batchnorm", epochs=20, batch_size=64)

# --- Experimento 1.8: Dropout ---
print("\n### Experimento 1.8: Con Dropout(0.3)")
model_dropout = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(128, activation='relu'), layers.Dropout(0.3),
    layers.Dense(128, activation='relu'), layers.Dropout(0.3),
    layers.Dense(128, activation='relu'), layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])
model_dropout.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_dropout = run_experiment(model_dropout, "08_dropout", epochs=20, batch_size=64)

# --- Experimento 1.9: Regularización L2 ---
print("\n### Experimento 1.9: Regularización L2 (1e-4)")
model_l2 = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dense(len(class_names), activation='softmax')
])
model_l2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_l2 = run_experiment(model_l2, "09_l2", epochs=20, batch_size=64)

# --- Experimento 1.10: Inicializador He Normal ---
print("\n### Experimento 1.10: Inicializador He Normal")
model_he = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    layers.Dense(len(class_names), activation='softmax')
])
model_he.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_he = run_experiment(model_he, "10_he_normal", epochs=20, batch_size=64)

# --- Experimento 1.11: Combinación óptima ---
print("\n### Experimento 1.11: Combinación de técnicas (BatchNorm + Dropout + L2 + He)")
model_combined = keras.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    
    layers.Dense(256, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    layers.Dense(128, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    
    layers.Dense(64, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    
    layers.Dense(len(class_names), activation='softmax')
])
model_combined.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_combined = run_experiment(model_combined, "11_combined", epochs=20, batch_size=64)

# Comparar arquitecturas
architecture_results = [
    result_baseline, result_deep, result_wide, result_pyramid,
    result_gelu, result_tanh, result_bn, result_dropout,
    result_l2, result_he, result_combined
]

print("\n" + "="*80)
print("📊 COMPARACIÓN DE ARQUITECTURAS")
print("="*80)
df_arch = compare_experiments(architecture_results)
print(df_arch.to_string(index=False))

# ============================================================================
# PARTE 2: EXPERIMENTOS CON OPTIMIZADORES
# ============================================================================

print("\n" + "="*80)
print("⚙️ PARTE 2: EXPERIMENTACIÓN CON OPTIMIZADORES")
print("="*80)

def create_standard_model():
    """Modelo estándar para comparar optimizadores."""
    return keras.Sequential([
        layers.Input(shape=(x_train.shape[1],)),
        layers.Dense(256, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        layers.Dense(128, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        layers.Dense(len(class_names), activation='softmax')
    ])

# --- Experimento 2.1: Adam LR=0.01 ---
print("\n### Experimento 2.1: Adam con LR=0.01")
model = create_standard_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_adam_01 = run_experiment(model, "21_adam_lr_0.01", epochs=20, batch_size=64)

# --- Experimento 2.2: Adam LR=0.001 ---
print("\n### Experimento 2.2: Adam con LR=0.001 (default)")
model = create_standard_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_adam_001 = run_experiment(model, "22_adam_lr_0.001", epochs=20, batch_size=64)

# --- Experimento 2.3: Adam LR=0.0001 ---
print("\n### Experimento 2.3: Adam con LR=0.0001")
model = create_standard_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_adam_0001 = run_experiment(model, "23_adam_lr_0.0001", epochs=20, batch_size=64)

# --- Experimento 2.4: SGD con momentum ---
print("\n### Experimento 2.4: SGD con momentum=0.9")
model = create_standard_model()
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_sgd = run_experiment(model, "24_sgd_momentum", epochs=20, batch_size=64)

# --- Experimento 2.5: SGD Nesterov ---
print("\n### Experimento 2.5: SGD con Nesterov momentum")
model = create_standard_model()
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_sgd_nesterov = run_experiment(model, "25_sgd_nesterov", epochs=20, batch_size=64)

# --- Experimento 2.6: RMSprop ---
print("\n### Experimento 2.6: RMSprop")
model = create_standard_model()
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_rmsprop = run_experiment(model, "26_rmsprop", epochs=20, batch_size=64)

# --- Experimento 2.7: AdamW ---
print("\n### Experimento 2.7: AdamW con weight_decay=1e-4")
model = create_standard_model()
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_adamw = run_experiment(model, "27_adamw", epochs=20, batch_size=64)

# --- Experimento 2.8: Batch size 32 ---
print("\n### Experimento 2.8: Batch size 32")
model = create_standard_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_batch32 = run_experiment(model, "28_batch32", epochs=20, batch_size=32)

# --- Experimento 2.9: Batch size 128 ---
print("\n### Experimento 2.9: Batch size 128")
model = create_standard_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
result_batch128 = run_experiment(model, "29_batch128", epochs=20, batch_size=128)

# Comparar optimizadores
optimizer_results = [
    result_adam_01, result_adam_001, result_adam_0001,
    result_sgd, result_sgd_nesterov, result_rmsprop, result_adamw,
    result_batch32, result_batch128
]

print("\n" + "="*80)
print("📊 COMPARACIÓN DE OPTIMIZADORES")
print("="*80)
df_opt = compare_experiments(optimizer_results)
print(df_opt.to_string(index=False))

# ============================================================================
# PARTE 3: EXPERIMENTOS CON CALLBACKS
# ============================================================================

print("\n" + "="*80)
print("⏱️ PARTE 3: EXPERIMENTACIÓN CON CALLBACKS")
print("="*80)

# --- Experimento 3.1: EarlyStopping ---
print("\n### Experimento 3.1: EarlyStopping")
model = create_standard_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, 
                                     restore_best_weights=True, verbose=1)
result_early = run_experiment(model, "31_early_stopping", epochs=50, batch_size=64, 
                              use_callbacks=[early_stop])

# --- Experimento 3.2: ReduceLROnPlateau ---
print("\n### Experimento 3.2: ReduceLROnPlateau")
model = create_standard_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                       patience=3, min_lr=1e-6, verbose=1)
result_reduce = run_experiment(model, "32_reduce_lr", epochs=30, batch_size=64, 
                               use_callbacks=[reduce_lr])

# --- Experimento 3.3: ModelCheckpoint ---
print("\n### Experimento 3.3: ModelCheckpoint")
model = create_standard_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint = callbacks.ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, "best_model_33.keras"),
    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)
result_checkpoint = run_experiment(model, "33_checkpoint", epochs=20, batch_size=64, 
                                   use_callbacks=[checkpoint])

# --- Experimento 3.4: Cosine Decay LR ---
print("\n### Experimento 3.4: Cosine Decay Learning Rate")
def cosine_decay(epoch, lr, initial_lr=0.001, min_lr=1e-6, epochs=30):
    if epoch == 0:
        return initial_lr
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / epochs))

model = create_standard_model()
initial_lr = 0.001
model.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_lr), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lr_scheduler = callbacks.LearningRateScheduler(
    lambda epoch, lr: cosine_decay(epoch, lr, initial_lr=initial_lr), verbose=1
)
result_cosine = run_experiment(model, "34_cosine_decay", epochs=30, batch_size=64, 
                               use_callbacks=[lr_scheduler])

# --- Experimento 3.5: Combinación de callbacks ---
print("\n### Experimento 3.5: Combinación de callbacks")
model = create_standard_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, 
                                     restore_best_weights=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                       patience=4, min_lr=1e-6, verbose=1)
checkpoint = callbacks.ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, "best_model_35.keras"),
    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)

result_combined_cb = run_experiment(model, "35_combined_callbacks", epochs=50, 
                                    batch_size=64, 
                                    use_callbacks=[early_stop, reduce_lr, checkpoint])

# Comparar callbacks
callback_results = [result_early, result_reduce, result_checkpoint, 
                    result_cosine, result_combined_cb]

print("\n" + "="*80)
print("📊 COMPARACIÓN DE CALLBACKS")
print("="*80)
df_cb = compare_experiments(callback_results)
print(df_cb.to_string(index=False))

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("🏆 RESUMEN FINAL - TODOS LOS EXPERIMENTOS")
print("="*80)

all_results = architecture_results + optimizer_results + callback_results
df_all = compare_experiments(all_results)

print("\n🥇 TOP 10 EXPERIMENTOS:")
print(df_all.head(10).to_string(index=False))

print("\n📉 BOTTOM 5 EXPERIMENTOS:")
print(df_all.tail(5).to_string(index=False))

# Guardar resultados
df_all.to_csv('experiment_results_summary.csv', index=False)
print("\n✅ Resultados guardados en 'experiment_results_summary.csv'")

# Mejor modelo
best = df_all.iloc[0]
print(f"\n🌟 MEJOR MODELO: {best['Experimento']}")
print(f"   Test Accuracy: {best['Test Acc']:.4f}")
print(f"   Val Accuracy:  {best['Val Acc']:.4f}")
print(f"   Parámetros:    {best['Parámetros']:,}")
print(f"   Tiempo:        {best['Tiempo (s)']:.1f}s")

print("\n" + "="*80)
print("✨ EXPERIMENTACIÓN COMPLETADA")
print("="*80)
print(f"\nTotal de experimentos: {len(all_results)}")
print(f"Ver logs en TensorBoard: tensorboard --logdir {ROOT_LOGDIR}")

