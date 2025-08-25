import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Function to define callbacks
def get_callbacks(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    return [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=os.path.join(output_dir, "best_model.h5"),
                        monitor="val_loss",
                        save_best_only=True,
                        verbose=1)
    ]

# Function to train the model
def train_model(model, train_gen, test_gen, output_dir="outputs", epochs=100):
    callbacks = get_callbacks(output_dir)

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save the trained model
    model.save(os.path.join(output_dir, "final_model.h5"))

    # Plot training and validation curves
    plot_training_curves(history, output_dir)
    return history

# Function to plot accuracy and loss curves
def plot_training_curves(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Loss plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label="Training Accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Save plots
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()
