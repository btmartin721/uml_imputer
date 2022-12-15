import sys

import numpy as np
import pandas as pd
import tensorflow as tf

class AutoEncoderCallbacks(tf.keras.callbacks.Callback):
    """Custom callbacks to use with subclassed autoencoder Keras model.

    Requires y, missing_mask, and sample_weight to be input variables to be properties with setters in the subclassed model.
    """

    def __init__(self):
        self.indices = None

    def on_epoch_begin(self, epoch, logs=None):
        """Shuffle input and target at start of epoch."""
        y = self.model.y.copy()
        missing_mask = self.model.missing_mask
        sample_weight = self.model.sample_weight

        n_samples = len(y)
        self.indices = np.arange(n_samples)
        np.random.shuffle(self.indices)

        self.model.y = y[self.indices]
        self.model.missing_mask = missing_mask[self.indices]

        if sample_weight is not None:
            self.model.sample_weight = sample_weight[self.indices]

    def on_train_batch_begin(self, batch, logs=None):
        """Get batch index."""
        self.model.batch_idx = batch

    def on_epoch_end(self, epoch, logs=None):
        """Unsort the row indices."""
        unshuffled = np.argsort(self.indices)

        self.model.y = self.model.y[unshuffled]
        self.model.missing_mask = self.model.missing_mask[unshuffled]

        if self.model.sample_weight is not None:
            self.model.sample_weight = self.model.sample_weight[unshuffled]


class UBPCallbacks(tf.keras.callbacks.Callback):
    """Custom callbacks to use with subclassed NLPCA/ UBP Keras models.

    Requires y, missing_mask, V_latent, and sample_weight to be input variables to be properties with setters in the subclassed model.
    """

    def __init__(self):
        self.indices = None

    def on_epoch_begin(self, epoch, logs=None):
        """Shuffle input and target at start of epoch."""
        y = self.model.y.copy()
        missing_mask = self.model.missing_mask
        sample_weight = self.model.sample_weight

        n_samples = len(y)
        self.indices = np.arange(n_samples)
        np.random.shuffle(self.indices)

        self.model.y = y[self.indices]
        self.model.V_latent = self.model.V_latent[self.indices]
        self.model.missing_mask = missing_mask[self.indices]

        if sample_weight is not None:
            self.model.sample_weight = sample_weight[self.indices]

    def on_train_batch_begin(self, batch, logs=None):
        """Get batch index."""
        self.model.batch_idx = batch

    def on_epoch_end(self, epoch, logs=None):
        """Unsort the row indices."""
        unshuffled = np.argsort(self.indices)

        self.model.y = self.model.y[unshuffled]
        self.model.V_latent = self.model.V_latent[unshuffled]
        self.model.missing_mask = self.model.missing_mask[unshuffled]

        if self.model.sample_weight is not None:
            self.model.sample_weight = self.model.sample_weight[unshuffled]


class UBPEarlyStopping(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Args:
        patience (int, optional): Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops. Defaults to 0.

        phase (int, optional): Current UBP Phase. Defaults to 3.
    """

    def __init__(self, patience=0, phase=3):
        super(UBPEarlyStopping, self).__init__()
        self.patience = patience
        self.phase = phase

        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

        # In UBP, the input gets refined during training.
        # So we have to revert it too.
        self.best_input = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()

            if self.phase != 2:
                # Only refine input in phase 2.
                self.best_input = self.model.V_latent
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

                if self.phase != 2:
                    self.model.V_latent = self.best_input
