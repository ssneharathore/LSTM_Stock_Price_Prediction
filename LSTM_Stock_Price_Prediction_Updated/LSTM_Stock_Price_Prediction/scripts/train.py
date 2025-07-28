from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, X_train, y_train):
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[early_stop])
    return history