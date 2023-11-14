print("Importing from 'models.py'")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations

class CVAE(keras.Model):
    '''
    Creates fully supervised CVAE Class 
    Training ar§chitecture: input -> latent space μ representation -> Proj(μ) -> contrastive loss
    '''
    def __init__(self,
            contrastive_loss,
            temp=0.07,
            latent_dim=2,
            layers_number=2,
            layers_size=[64, 32],
            layer_size_projection=16,
            **kwargs):
        super().__init__(**kwargs)
        self.encoder = build_encoder(latent_dim, layers_number, layers_size)
        self.projection_head = build_projection_head(latent_dim, layer_size_projection)
        self.temperature = temp
        self.contrastive_loss_fn = contrastive_loss
        self.contrastive_loss_tracker = keras.metrics.Mean(name="contrastive_tracker")

    @property
    def metrics(self):
        return [self.contrastive_loss_tracker]

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            # Forward pass to create reconstruction + computes loss
            data, labels = inputs
            z_mean, z_log_var, z = self.encoder(data, training=True)
            projection = self.projection_head(z, training=True)
            contrastive_loss = self.contrastive_loss_fn(projection, labels=labels, temperature=self.temperature)

        # Apply gradients and update losses
        grads = tape.gradient(contrastive_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {"contrastive_loss": self.contrastive_loss_tracker.result()}

    def test_step(self, data):
        # Unpacks the data
        features, labels = data
        # Computes latent space representation and loss
        z_mean, z_log_var, z = self.encoder(features, training=False)
        projection = self.projection_head(z, training=False)
        valid_loss = self.contrastive_loss_fn(projection, labels=labels, temperature=self.temperature)

        # Updates loss metrics
        for metric in self.metrics:
            if metric.name != "contrastive_loss":
                metric.update_state(valid_loss)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        _, _, encoded = self.encoder(inputs, training=False)
        projected = self.projection_head(encoded, training=False)
        return projected

def build_encoder(latent_dim, layers_number, layers_size):
    '''
    Encoder as defined in gitlab: https://gitlab.cern.ch/cms-l1-ad/l1_anomaly_ae
    '''
    enc_inputs = keras.Input(shape=(57,))
    x = layers.Dense(32)(enc_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    for l in range(layers_number):
        x = layers.Dense(layers_size[l])(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
    
    # If encoer is used for sampling, forms gaussian and returns μ, σ, and sampled point
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sample_Layer()([z_mean, z_log_var])
    encoder = keras.Model(enc_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

def build_projection_head(latent_dim, layer_size_projection):
    '''
    Build MLP projection head before computing loss as suggested by https://arxiv.org/pdf/2002.05709.pdf
    Disregarded after training
    '''
    projection_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(layer_size_projection)(projection_inputs)
    x = layers.LeakyReLU()(x)
    projection = layers.Dense(latent_dim)(x)
    projection_head = keras.Model(projection_inputs, projection, name="projection_head")
    return projection_head

class Sample_Layer(layers.Layer):
    '''
    Builds custom sampling layer from gaussian distribution for VAE reconstruction 
    '''
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dims = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dims))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
    
    