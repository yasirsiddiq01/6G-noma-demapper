"""
SICNet Neural Network Model
Implements the neural network architecture for learned interference cancellation,
inspired by CTTC's research on AI for NOMA systems.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


def build_sicnet(input_dim=4, output_dim=2, hidden_units=[128, 256, 128], dropout_rate=0.2):
    """
    Build SICNet model - a neural network that learns to demap NOMA signals.
    
    Architecture:
        Input (4 features) → Dense(128) → BatchNorm → Dropout
                          → Dense(256) → BatchNorm → Dropout
                          → Dense(128) → BatchNorm
                          → Two output heads (User 1 and User 2)
    
    Args:
        input_dim: Number of input features (default: 4)
                  [received_real, received_imag, |h1|, |h2|]
        output_dim: Number of output bits per user (default: 2 for QPSK)
        hidden_units: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        model: Compiled Keras model (if compiled=False)
        or compiled model ready for training
    """
    inputs = layers.Input(shape=(input_dim,), name='input')
    
    # Shared feature extraction layers
    x = layers.Dense(hidden_units[0], activation='relu', name='dense1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(dropout_rate, name='dropout1')(x)
    
    x = layers.Dense(hidden_units[1], activation='relu', name='dense2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Dropout(dropout_rate, name='dropout2')(x)
    
    x = layers.Dense(hidden_units[2], activation='relu', name='dense3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    
    # Two output heads (one for each user)
    user1_output = layers.Dense(output_dim, activation='sigmoid', name='user1')(x)
    user2_output = layers.Dense(output_dim, activation='sigmoid', name='user2')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=[user1_output, user2_output], 
                        name='SICNet')
    
    return model


def build_attention_sicnet(input_dim=4, output_dim=2):
    """
    Advanced SICNet with attention mechanism (closer to CTTC's HELENA project).
    
    Adds self-attention to help the network focus on relevant features.
    """
    inputs = layers.Input(shape=(input_dim,), name='input')
    
    # Initial dense layers
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Reshape for attention (add sequence dimension)
    x_seq = layers.Reshape((1, 128))(x)
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=32
    )(x_seq, x_seq)
    
    # Add & Norm (residual connection)
    x_seq = layers.Add()([x_seq, attention_output])
    x_seq = layers.LayerNormalization()(x_seq)
    
    # Feed-forward network
    ff = layers.Dense(256, activation='relu')(x_seq)
    ff = layers.Dense(128)(ff)
    
    # Add & Norm
    x_seq = layers.Add()([x_seq, ff])
    x_seq = layers.LayerNormalization()(x_seq)
    
    # Back to original shape
    x = layers.Flatten()(x_seq)
    x = layers.Dropout(0.2)(x)
    
    # Output layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    user1_output = layers.Dense(output_dim, activation='sigmoid', name='user1')(x)
    user2_output = layers.Dense(output_dim, activation='sigmoid', name='user2')(x)
    
    model = models.Model(inputs=inputs, outputs=[user1_output, user2_output],
                        name='AttentionSICNet')
    
    return model


def build_ensemble_sicnet(input_dim=4, output_dim=2, n_models=3):
    """
    Build an ensemble of SICNet models for improved performance.
    
    Uses multiple models with different initializations and averages their predictions.
    """
    inputs = layers.Input(shape=(input_dim,), name='input')
    
    all_user1_outputs = []
    all_user2_outputs = []
    
    for i in range(n_models):
        # Create a sub-model with slightly different architecture
        x = layers.Dense(128, activation='relu', name=f'dense1_{i}')(inputs)
        x = layers.BatchNormalization(name=f'bn1_{i}')(x)
        x = layers.Dropout(0.2 + 0.1 * i, name=f'dropout1_{i}')(x)
        
        x = layers.Dense(256, activation='relu', name=f'dense2_{i}')(x)
        x = layers.BatchNormalization(name=f'bn2_{i}')(x)
        x = layers.Dropout(0.2 + 0.1 * i, name=f'dropout2_{i}')(x)
        
        x = layers.Dense(128, activation='relu', name=f'dense3_{i}')(x)
        x = layers.BatchNormalization(name=f'bn3_{i}')(x)
        
        user1_out = layers.Dense(output_dim, activation='sigmoid', name=f'user1_{i}')(x)
        user2_out = layers.Dense(output_dim, activation='sigmoid', name=f'user2_{i}')(x)
        
        all_user1_outputs.append(user1_out)
        all_user2_outputs.append(user2_out)
    
    # Average the outputs
    avg_user1 = layers.Average(name='user1')(all_user1_outputs)
    avg_user2 = layers.Average(name='user2')(all_user2_outputs)
    
    model = models.Model(inputs=inputs, outputs=[avg_user1, avg_user2],
                        name='EnsembleSICNet')
    
    return model


def compile_sicnet(model, learning_rate=0.001):
    """
    Compile SICNet model with appropriate loss and metrics.
    
    Args:
        model: Uncompiled Keras model
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'user1': 'binary_crossentropy',
            'user2': 'binary_crossentropy'
        },
        metrics={
            'user1': ['accuracy'],
            'user2': ['accuracy']
        }
    )
    return model


def get_callbacks(patience=10, model_path='best_model.h5'):
    """
    Get training callbacks for SICNet.
    
    Args:
        patience: Patience for early stopping
        model_path: Path to save best model
        
    Returns:
        List of callbacks
    """
    callbacks = [
        # Reduce learning rate when validation loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Stop training when no improvement
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Log training progress
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    return callbacks


if __name__ == "__main__":
    # Test model creation
    model = build_sicnet()
    model = compile_sicnet(model)
    model.summary()
    
    print("\n✅ SICNet model created successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shapes: {[out.shape for out in model.outputs]}")
    
    # Test with random data
    x_test = np.random.randn(32, 4)
    y1_test = np.random.randint(0, 2, (32, 2))
    y2_test = np.random.randint(0, 2, (32, 2))
    
    # Forward pass
    y1_pred, y2_pred = model.predict(x_test)
    print(f"\nForward pass successful!")
    print(f"Predictions shape: {y1_pred.shape}, {y2_pred.shape}")
