# train.py
# Train CNN on GTSRB dataset with image processing

import os
from preprocessing import load_train_set, train_val_split, IMG_SIZE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



MODEL_PATH = "model/traffic_sign_cnn.h5"
NUM_CLASSES = 43

def build_model(input_shape=(32,32,3), num_classes=NUM_CLASSES):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3,3), activation="relu", padding="same"),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3,3), activation="relu", padding="same"),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128, (3,3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    os.makedirs("model", exist_ok=True)
    print("[info] Loading training data with image processing...")
    X, y = load_train_set("dataset/Train")  # uses preprocessing.py with image processing
    X_train, X_val, y_train, y_val = train_val_split(X, y, val_size=0.2)

    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=128,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    model.save(MODEL_PATH)
    print(f"[info] Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
