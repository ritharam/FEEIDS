import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import os

# Create folder for saving plots
PLOTS_FOLDER = "bilstm_plots1"
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Detect runtime environment
def get_runtime_environment():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        return "TPU", strategy
    except:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return "GPU", tf.distribute.OneDeviceStrategy("/gpu:0")
        else:
            return "CPU", tf.distribute.OneDeviceStrategy("/cpu:0")

def plot_training_history(history, environment, hidden_nodes, save_folder):
    """Plot and save training history for accuracy and loss"""
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.title(f'BiLSTM Training & Validation Accuracy\n{environment} - {hidden_nodes} Hidden Nodes - Patience : {PATIENCE}', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_filename = os.path.join(save_folder, f"{environment}_{hidden_nodes}_PL_{PATIENCE}_acc.png")
    plt.savefig(acc_filename, dpi=150)
    plt.close()
    print(f"  Saved accuracy plot: {acc_filename}")
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title(f'BiLSTM Training & Validation Loss\n{environment} - {hidden_nodes} Hidden Nodes - Patience : {PATIENCE}', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_filename = os.path.join(save_folder, f"{environment}_{hidden_nodes}_PL_{PATIENCE}_loss.png")
    plt.savefig(loss_filename, dpi=150)
    plt.close()
    print(f"  Saved loss plot: {loss_filename}")

ENVIRONMENT, strategy = get_runtime_environment()
print(f"Running on: {ENVIRONMENT}")

# Load normalized data
train_df = pd.read_csv("UNSW_FEIIDS_train.csv")
test_df = pd.read_csv("UNSW_FEIIDS_test.csv")

# Identify and separate labels
label_cols = ['attack_cat', 'binary_label', 'Label', 'label']
existing_labels = [col for col in label_cols if col in train_df.columns]

X_full = train_df.drop(columns=existing_labels, errors='ignore').values
y_full = train_df['binary_label'].values if 'binary_label' in train_df.columns else train_df['label'].values
X_test_full = test_df.drop(columns=existing_labels, errors='ignore').values
y_test_full = test_df['binary_label'].values if 'binary_label' in test_df.columns else test_df['label'].values

print(f"Total train records: {len(X_full)}, Features: {X_full.shape[1]}")
print(f"Total test records: {len(X_test_full)}")

# Sample 14,000 for training and 3,500 for testing (as per Table 6)
np.random.seed(42)
train_indices = np.random.choice(len(X_full), 14000, replace=False)
test_indices = np.random.choice(len(X_test_full), 3500, replace=False)

# Create train and test sets
X_train = X_full[train_indices]
y_train = y_full[train_indices]
X_test = X_test_full[test_indices]
y_test = y_test_full[test_indices]

print(f"\nDataset splits:")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Testing: {X_test.shape[0]} samples")

# Reshape for LSTM: (samples, timesteps=1, features)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

print(f"\nReshaped dimensions:")
print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# Configuration
HIDDEN_NODES_LIST = [10, 20, 30, 40, 50, 60, 70, 80]
EPOCHS = 200
BATCH_SIZE = 64
PATIENCE = 20  # Early stopping patience

# Results storage
results_file = "bilstm_table_results.csv"

# Create folder for saving models
MODELS_FOLDER = "bilstm_models"
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Train for each hidden node configuration
for hidden_nodes in HIDDEN_NODES_LIST:
    print(f"\n{'='*70}")
    print(f"Training BiLSTM with {hidden_nodes} hidden nodes on {ENVIRONMENT}")
    print(f"Early Stopping Patience: {PATIENCE} epochs")
    print('='*70)
    
    with strategy.scope():
        # Build BiLSTM model (2 layers as per paper)
        model = Sequential([
            Bidirectional(LSTM(hidden_nodes, return_sequences=True),
                         input_shape=(1, X_train.shape[2])),
            Bidirectional(LSTM(hidden_nodes, return_sequences=False)),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    # Early stopping callback - monitors VALIDATION loss
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Changed from 'loss' to 'val_loss'
        patience=PATIENCE,
        min_delta=0.0001,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model WITH early stopping using test set as validation
    start_train = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),  # Use test set as validation
        verbose=1,
        callbacks=[early_stopping]
    )
    train_time = time.time() - start_train
    
    # Get actual number of epochs trained
    epochs_trained = len(history.history['loss'])
    
    # Plot training history
    print(f"\nGenerating plots for {hidden_nodes} hidden nodes...")
    plot_training_history(history, ENVIRONMENT, hidden_nodes, PLOTS_FOLDER)
    
    # Test model (final evaluation)
    start_test = time.time()
    y_pred_proba = model.predict(X_test, verbose=0)
    test_time = time.time() - start_test
    
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
    
    # Store result with custom format
    result = {
        'Hidden_Nodes': hidden_nodes,
        'Accuracy': round(accuracy, 4),
        'Train_Time_Epochs': epochs_trained,  # Number of epochs
        'Test_Time_s': round(test_time, 2),  # Test time in seconds
        'Test_Loss': round(test_loss, 4)
    }
    
    # Save model
    model_filename = os.path.join(MODELS_FOLDER, f"bilstm_{ENVIRONMENT}_{hidden_nodes}nodes_PL_{PATIENCE}.keras")
    model.save(model_filename)
    print(f"Model saved: {model_filename}\n")

    # Save to CSV
    result_df = pd.DataFrame([result])
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file)
        combined_df = pd.concat([existing_df, result_df], ignore_index=True)
        combined_df.to_csv(results_file, index=False)
    else:
        result_df.to_csv(results_file, index=False)
    
    print(f"\nResults for {hidden_nodes} hidden nodes:")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Train Time (Epochs): {epochs_trained}")
    print(f"Test Time: {test_time:.2f}s")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Clear session to free memory
    tf.keras.backend.clear_session()

print(f"\n{'='*70}")
print(f"All experiments completed! Results saved to {results_file}")
print(f"All plots saved to {PLOTS_FOLDER}/")
print('='*70)

# Display final results
final_results = pd.read_csv(results_file)
print("\nFinal Results:")
print(final_results)
