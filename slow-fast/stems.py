import tensorflow
from tensorflow.keras import layers

class BaseStem(layers.Layer):
    def __init__(self, dim_in, dim_out, kernel, stride, padding, epsilon, bn_moment, normalization_method=tensorflow.keras.layers.BatchNormalization):
        super(BaseStem,self).__init__()
        if padding is not None:
            self.conv_pad = tensorflow.keras.layers.ZeroPadding3D(padding=padding)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.epsilon = epsilon
        self.bn_moment = bn_moment
        if self.padding is not None:
          self.conv_pad = tensorflow.keras.layers.ZeroPadding3D(padding=padding)
        self.conv = tensorflow.keras.layers.Conv3D(self.dim_out, self.kernel, strides=self.stride, padding='valid', use_bias=False)
        self.bn = tensorflow.keras.layers.BatchNormalization(axis=-1, epsilon=self.epsilon, momentum=self.bn_moment)
        self.relu = tensorflow.keras.layers.ReLU()
        self.pool_pad = tensorflow.keras.layers.ZeroPadding3D(padding=[0,1,1])
        self.pool = tensorflow.keras.layers.MaxPooling3D(pool_size=(1,3,3), strides=(1,2,2))

    def call(self, inputs, training=None):
        x = tensorflow.identity(inputs)
        if hasattr(self, "conv_pad"):
            x = self.conv_pad(x)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.pool_pad(x)
        
        return self.pool(x)
    def compute_output_shape(self, input_shape):
      return self.pool.output_shape
class InitialStem(layers.Layer):
    def __init__(self, dim_in, dim_out, kernel, stride, padding, normalization_method=tensorflow.keras.layers.BatchNormalization, epsilon=1e-5, bn_mom=0.1):
        super(InitialStem,self).__init__()
        # dim_in and dim_out is a list to allow us to operate on multiple pathways
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.epsilon = epsilon
        self.bn_mom = bn_mom
        self.normalization_method = normalization_method

    def build(self, input_shape):
        self.layers = {}
        l = BaseStem(self.dim_in, self.dim_out, self.kernel, self.stride, self.padding, self.epsilon, self.bn_mom, self.normalization_method) 
        self.layers = l
        super(InitialStem, self).build(input_shape)
    def call(self, inputs, training=None):
        l = self.layers
        
        return l(inputs, training=training)
    def compute_output_shape(self, input_shape):
        return self.layers.output_shape

class FinalStem(layers.Layer):
    def __init__(self, dim_in, classes, pool, dropout, pathways, activation="softmax"):
        super(FinalStem, self).__init__()

        self.number_of_pathways = pathways
        self.pools=[]
        for p in range(self.number_of_pathways):
            pl = tensorflow.keras.layers.AveragePooling3D(pool_size=pool[p], strides=1)
            self.pools.append(pl)

        if dropout is not None and dropout > 0.0:
            self.drop = tensorflow.keras.layers.Dropout(dropout)
        self.concat = tensorflow.keras.layers.Concatenate(axis=-1)
        self.flatten = tensorflow.keras.layers.Flatten()
        self.fc = tensorflow.keras.layers.Dense(classes, use_bias=True, activation=activation)


    def call(self,inputs, training=None):
        ps=[]
        for p in range(self.number_of_pathways):
            pl = self.pools[p](inputs[p])
            ps.append(pl)
        x = self.concat(ps) if self.number_of_pathways > 1 else ps[0]  

        if hasattr(self, "drop") and training:
            x = self.drop(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def compute_output_shape(self, input_shape):
        return self.fc.output_shape