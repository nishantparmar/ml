import tensorflow as tf


# Neural Network model
class CartPoleNN(tf.keras.Model):
    def __init__(self, num_actions):
        super(CartPoleNN, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(16, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, inputs):
        x = self.hidden_layer(inputs)
        return self.output_layer(x)
