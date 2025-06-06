import tensorflow as tf
from tensorflow.keras import layers, Model
class CBHG(tf.keras.layers.Layer):
    def __init__(self, conv_bank_filters, proj_filters, highway_units, gru_units, **kwargs):
        super(CBHG, self).__init__(**kwargs)
        self.conv_bank_filters = conv_bank_filters
        self.proj_filters = proj_filters
        self.highway_units = highway_units
        self.gru_units = gru_units

        self.conv_bank = [layers.Conv1D(filters=conv_bank_filters, kernel_size=k, padding='same', activation='relu')
                          for k in range(1, 9)]
        self.batch_norms = [layers.BatchNormalization() for _ in self.conv_bank]

        self.max_pool = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')

        self.proj1 = layers.Conv1D(filters=proj_filters[0], kernel_size=3, padding='same', activation='relu')
        self.proj2 = layers.Conv1D(filters=proj_filters[1], kernel_size=3, padding='same')

        self.highway_layers = [layers.Dense(highway_units, activation='relu') for _ in range(4)]

        self.bi_gru = layers.Bidirectional(layers.GRU(gru_units, return_sequences=True))

    def call(self, inputs, training=False):
        x = inputs
        conv_outputs = []
        for conv, bn in zip(self.conv_bank, self.batch_norms):
            c = conv(x)
            c = bn(c, training=training)
            conv_outputs.append(c)

        x = tf.concat(conv_outputs, axis=-1)
        x = self.max_pool(x)
        x = self.proj1(x)
        x = self.proj2(x)

        if x.shape[-1] == inputs.shape[-1]:
            x += inputs

        if x.shape[-1] != self.highway_units:
            x = layers.Dense(self.highway_units)(x)

        for layer in self.highway_layers:
            x = layer(x)

        x = self.bi_gru(x)
        return x

    def get_config(self):
        config = super(CBHG, self).get_config()
        config.update({
            "conv_bank_filters": self.conv_bank_filters,
            "proj_filters": self.proj_filters,
            "highway_units": self.highway_units,
            "gru_units": self.gru_units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ========== CBHG Model ==========

class CBHGModel(tf.keras.Model):
    def __init__(self, vocab_size, label_size, embedding_dim=128, conv_bank_filters=128,
                 proj_filters=[128, 128], highway_units=128, gru_units=128, **kwargs):
        super(CBHGModel, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.embedding_dim = embedding_dim
        self.conv_bank_filters = conv_bank_filters
        self.proj_filters = proj_filters
        self.highway_units = highway_units
        self.gru_units = gru_units

        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.cbhg = CBHG(conv_bank_filters, proj_filters, highway_units, gru_units)
        self.classifier = layers.Dense(label_size)

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.cbhg(x, training=training)
        logits = self.classifier(x)
        return logits

    def get_config(self):
        config = super(CBHGModel, self).get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "label_size": self.label_size,
            "embedding_dim": self.embedding_dim,
            "conv_bank_filters": self.conv_bank_filters,
            "proj_filters": self.proj_filters,
            "highway_units": self.highway_units,
            "gru_units": self.gru_units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)