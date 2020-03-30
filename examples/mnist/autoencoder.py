import tensorflow as tf


class Autoencoder:
    def __init__(self, encoder_name, decoder_name, autoencoder_name):
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.autoencoder_name = autoencoder_name
        self.encoder = None
        self.decoder = None
        self.autoencoder = None

    def build(self, input_dim, code_dim, hidden_dim):
        self.encoder = tf.keras.Sequential(name=self.encoder_name)
        self.decoder = tf.keras.Sequential(name=self.decoder_name)
        self.autoencoder = tf.keras.Sequential(name=self.autoencoder_name)

        self.encoder.add(tf.keras.layers.Input(shape=(input_dim,)))
        self.decoder.add(tf.keras.layers.Input(shape=(code_dim,)))
        self.autoencoder.add(tf.keras.layers.Input(shape=(input_dim,)))

        depth = len(hidden_dim)
        for i, j in zip(range(depth), reversed(range(depth))):
            self.encoder.add(tf.keras.layers.Dense(hidden_dim[i], activation='relu'))
            self.decoder.add(tf.keras.layers.Dense(hidden_dim[j], activation='relu'))
        self.encoder.add(tf.keras.layers.Dense(code_dim, activation='relu'))
        self.decoder.add(tf.keras.layers.Dense(input_dim, activation='sigmoid'))

        self.autoencoder.add(self.encoder)
        self.autoencoder.add(self.decoder)

    def compile(self, optimizer='adam', loss='mse'):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    def fit(self, x_train, y_train, epochs=100, batch_size=128, validation_data=None):
        history = self.autoencoder.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                       validation_data=validation_data)
        return history

    def save(self, encoder_fp=None, decoder_fp=None, autoencoder_fp=None):
        if encoder_fp is not None:
            self.encoder.save(encoder_fp)
        if decoder_fp is not None:
            self.decoder.save(decoder_fp)
        if autoencoder_fp is not None:
            self.autoencoder.save(autoencoder_fp)
