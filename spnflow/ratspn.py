import tensorflow as tf
from layers import InputLayer, ProductLayer, SumLayer


class RatSpn(tf.keras.Model):
    def __init__(self, n_classes, n_sum, n_distributions, rg_layers):
        super(RatSpn, self).__init__()
        self.n_classes = n_classes
        self.n_sum = n_sum
        self.n_features = len(rg_layers[-1][0])
        self.n_distributions = n_distributions
        self._rg_layers = rg_layers
        self._layers = []

    def input_shape(self):
        return self.n_features

    def output_shape(self):
        return self.n_classes

    def build(self, input_shape):
        # Add the input distributions layer
        input_layer = InputLayer(self._rg_layers[0], self.n_distributions)
        self._layers.append(input_layer)

        # Alternate between product and sum layers
        for i in range(1, len(self._rg_layers)):
            if i % 2 == 1:
                product_layer = ProductLayer()
                self._layers.append(product_layer)
            else:
                sum_layer = SumLayer(self.n_sum)
                self._layers.append(sum_layer)

        # Add the root sum layer
        root_layer = SumLayer(self.n_classes, name='root_sum_layer')
        self._layers.append(root_layer)

        # Call the parent class build method
        super(RatSpn, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    from spnflow.region import RegionGraph
    rg = RegionGraph([0, 1, 2, 3, 4, 5, 6, 7, 8])

    for i in range(3):
        rg.random_split(2)
    rg.make_layers()
    layers = rg.layers()
    for layer in reversed(layers):
        print(layer)

    spn = RatSpn(4, 2, 2, layers)
    spn.compile(optimizer='adam')

