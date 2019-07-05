import tensorflow as tf


class Model(tf.keras.Model):

    def __init__(self, num_classes=1):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.dense_1 = tf.layers.Dense(30, activation='relu')
        self.dense_2 = tf.layers.Dense(60, activation='relu')
        self.dense_3 = tf.layers.Dense(60, activation='relu')
        self.dense_4 = tf.layers.Dense(30, activation='relu')
        self.dense_5 = tf.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = tf.layers.dropout(x)
        x = self.dense_2(x)
        x = tf.layers.dropout(x)
        x = self.dense_3(x)
        x = tf.layers.dropout(x)
        x = self.dense_4(x)
        x = tf.layers.dropout(x)
        x = self.dense_5(x)
        return x

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

