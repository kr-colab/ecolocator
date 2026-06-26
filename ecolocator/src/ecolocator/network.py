import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


@tf.keras.utils.register_keras_serializable(name="euclid_loss")
def euclid_loss(y_true, y_pred):
    """Euclidean distance between 2-D coordinates (location)."""
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def build_network(
    n_snps: int,
    num_covs: int,
    nlayers: int = 10,
    width: int = 256,
    dropout_prop: float = 0.25,
    loc_weight: float = 1.0,
    env_weight: float = 1.0,
) -> tf.keras.Model:
    geno_input = tf.keras.Input(shape=(n_snps,), name="geno_input")
    trunk = tf.keras.layers.BatchNormalization()(geno_input)
    for _ in range(int(np.floor(nlayers / 2))):
        trunk = tf.keras.layers.Dense(width, activation="elu")(trunk)
    trunk = tf.keras.layers.Dropout(dropout_prop)(trunk)
    for _ in range(int(np.ceil(nlayers / 2))):
        trunk = tf.keras.layers.Dense(width, activation="elu")(trunk)
    loc_output = tf.keras.layers.Dense(2)(tf.keras.layers.Dense(2)(trunk))
    env_output = tf.keras.layers.Dense(num_covs)(tf.keras.layers.Dense(num_covs)(trunk))
    model = tf.keras.Model(inputs=geno_input, outputs=[loc_output, env_output])
    model.compile(
        optimizer="Adam",
        loss=[euclid_loss, "mse"],
        loss_weights=[loc_weight, env_weight],
    )
    return model


def build_callbacks(
    patience: int = 100,
    verbose: int = 1,
) -> list:
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=patience,
        restore_best_weights=True,
    )
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=int(patience / 6),
        verbose=verbose,
        mode="auto",
        min_delta=0,
        cooldown=0,
        min_lr=0,
    )
    return [earlystop, reducelr]


def train_network(
    model: tf.keras.Model,
    traingen: np.ndarray,
    testgen: np.ndarray,
    trainlocs: list,
    testlocs: list,
    max_epochs: int = 5000,
    batch_size: int = 32,
    patience: int = 100,
    verbose: int = 1,
) -> tf.keras.callbacks.History:
    callbacks = build_callbacks(patience=patience, verbose=verbose)
    history = model.fit(
        traingen,
        trainlocs,
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=verbose,
        validation_data=(testgen, testlocs),
        callbacks=callbacks,
    )
    return history
