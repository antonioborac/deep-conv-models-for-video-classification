import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import Sequential

def get_t_fun(desired_func):
    t_functions = {"bottleneck":Bottleneck,"basic":Basic}

    return t_functions[desired_func]


class Basic(layers.Layer):
    """
    Basic transformation: Tx3x3, 1x3x3, where T is the size of temporal kernel.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner=None,
        num_groups=1,
        stride_1=None,
        relu_in=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=None
    ):
        super(Basic, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = relu_in
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._stride = stride

        self.a_pad = tensorflow.keras.layers.ZeroPadding3D(padding=[int(self.temp_kernel_size // 2), 1, 1])
        
        self.a = tensorflow.keras.layers.Conv3D(self._dim_out, kernel_size=[self.temp_kernel_size, 3, 3], strides=[1, self._stride, self._stride], padding='valid', dilation_rate=1, activation=None, use_bias=False) 
        
        self.a_bn = tensorflow.keras.layers.BatchNormalization(axis=-1, momentum=self._bn_mmt, epsilon=self._eps)
        self.a_relu = tensorflow.keras.layers.ReLU()
        
        self.b_pad = tensorflow.keras.layers.ZeroPadding3D(padding=[0, 1, 1])
        self.b = tensorflow.keras.layers.Conv3D(self._dim_out, kernel_size=[1, 3, 3], strides=[1, 1, 1], padding='valid', dilation_rate=1, activation=None, use_bias=False)
        
        self.b_bn = tensorflow.keras.layers.BatchNormalization(axis=-1, momentum=self._bn_mmt, epsilon=self._eps)
        
    def call(self, inputs, training=False):
        x = tensorflow.identity(inputs)
        x = self.a_pad(x)
        x = self.a(x)
        x = self.a_bn(x, training=training)
        x = self.a_relu(x)

        x = self.b_pad(x)
        x = self.b(x)
        x = self.b_bn(x, training=training)
        return x
    def compute_output_shape(self, input_shape):
      return self.b_bn.output_shape

class Bottleneck(layers.Layer):
    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1=False,
        relu_in=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
    ):
        super(Bottleneck, self).__init__()
        self._temp_kernel_size = temp_kernel_size
        self._inplace_relu = relu_in
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._stride = stride
        self._dim_inner = dim_inner
        self._num_groups = num_groups
        self._dilation = dilation

        (str1x1, str3x3) = (self._stride, 1) if self._stride_1x1 else (1, self._stride)

        # Tx1x1, BN, ReLU.
        self.a_pad = tensorflow.keras.layers.ZeroPadding3D(padding=[int(self._temp_kernel_size // 2), 0, 0])
        self.a = tensorflow.keras.layers.Conv3D(self._dim_inner, kernel_size=[self._temp_kernel_size, 1, 1], strides=[1, str1x1, str1x1], padding='valid', dilation_rate=1, activation=None, use_bias=False)#, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint 
        
        self.a_bn = tensorflow.keras.layers.BatchNormalization(axis=-1, momentum=self._bn_mmt, epsilon=self._eps)
        self.a_relu = tensorflow.keras.layers.ReLU()
        
        # 1x3x3, BN, ReLU.
        self.b_pad = tensorflow.keras.layers.ZeroPadding3D(padding=[0, self._dilation, self._dilation])
        self.b = tensorflow.keras.layers.Conv3D(self._dim_inner, kernel_size=[1, 3, 3], strides=[1, str3x3, str3x3], padding='valid',dilation_rate=[1,self._dilation,self._dilation], activation=None, use_bias=False)#, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint 
        #note that here should be used separableconv3d?
    
        self.b_bn = tensorflow.keras.layers.BatchNormalization(axis=-1, momentum=self._bn_mmt, epsilon=self._eps)
        
        self.b_relu = tensorflow.keras.layers.ReLU()

        # 1x1x1, BN.
        self.c = tensorflow.keras.layers.Conv3D(self._dim_out, kernel_size=[1, 1, 1], strides=[1, 1, 1], padding='valid', activation=None, use_bias=False)#, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint 
        
        self.c_bn = tensorflow.keras.layers.BatchNormalization(axis=-1, momentum=self._bn_mmt, epsilon=self._eps) 
        
    def call(self, inputs, training=False):
        x = tensorflow.identity(inputs)
        x = self.a_pad(x)
        x = self.a(x)
        x = self.a_bn(x, training=training)
        x = self.a_relu(x)

        x = self.b_pad(x)
        x = self.b(x)
        x = self.b_bn(x, training=training)
        x = self.b_relu(x)

        x = self.c(x)
        x = self.c_bn(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        return self.c_bn.output_shape
class ResnetBlock(tensorflow.keras.Model):
    def __init__(
        self,
        dim_in,
        dim_out,
        temporal_kernel_size,
        stride,
        t_fun,
        dim_inner,
        num_groups=1,
        stride_1=False,
        relu_in=True,
        eps=1e-5,
        bn_mnt=0.1,
        dilation=1,
        name=None
    ):
        super(ResnetBlock, self).__init__()
        self._relu_in = relu_in
        self._eps = eps
        self._bn_mnt = bn_mnt
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._temporal_kernel_size = temporal_kernel_size
        self._stride = stride
        self._t_fun = t_fun
        self._dim_inner = dim_inner
        self._num_groups = num_groups
        self._stride_1 = stride_1
        self._relu_in = relu_in
        self._dilation = dilation
        
        if(self._stride != 1) or (self._dim_in != self._dim_out):
            self.branch1 = tensorflow.keras.layers.Conv3D(self._dim_out, kernel_size=(1,1,1), strides=[1,self._stride,self._stride], padding='valid', dilation_rate=1, activation=None, use_bias=False)#, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint
            self.branch1_bn = tensorflow.keras.layers.BatchNormalization(axis=-1, momentum=self._bn_mnt, epsilon=self._eps)
            
        self.branch2 = self._t_fun(self._dim_in, self._dim_out, self._temporal_kernel_size, self._stride, self._dim_inner, self._num_groups, stride_1=self._stride_1, relu_in=self._relu_in, dilation=self._dilation)

        self.relu = tensorflow.keras.layers.ReLU()
    def call(self, inputs, training=False):
        x = tensorflow.identity(inputs)
        if hasattr(self, "branch1"):
            x = self.branch1_bn(self.branch1(x),training=training) + self.branch2(x, training=training)
        else:
            x = x + self.branch2(x, training=training)
        return self.relu(x)
    
    def compute_output_shape(self, input_shape):
      return self.relu.output_shape

def res_stage(dim_in=None, dim_out=None, dim_inner=None, temp_kernel_size=None, stride=None,  number_of_blocks=None,  num_groups=None, num_block_temp_kernel=None, inst="softmax", t_func=Bottleneck, dilation=None,stride_1=False, relu_in=True):
    temporal_kernel_size_f=(temp_kernel_size * number_of_blocks)[: num_block_temp_kernel] + [1] * (number_of_blocks - num_block_temp_kernel)
    
    model = Sequential()
    for i in range(number_of_blocks):
        tfun = t_func
        model.add(ResnetBlock(dim_in if i==0 else dim_out, dim_out,temporal_kernel_size_f[i], stride if i == 0 else 1, tfun, dim_inner, num_groups, stride_1, relu_in, dilation=dilation))    
    return model
class ResStage(tensorflow.keras.Model):
    def __init__(self, dim_in, dim_out, stride, temp_kernel_size, number_of_blocks, dim_inner, num_groups, num_block_temp_kernel, dilation, inst="softmax", t_func="bottleneck",stride_1=False, relu_in=True):
        super(ResStage,self).__init__()
        self._number_of_blocks = number_of_blocks
        
        self._temporal_kernel_size=(temp_kernel_size * number_of_blocks)[: num_block_temp_kernel] + [1] * (number_of_blocks - num_block_temp_kernel)
        
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._stride = stride
        self._dim_inner = dim_inner
        self._num_groups = num_groups
        self._t_func = t_func
        self._stride_1 = stride_1
        self._relu_in = relu_in
        self._inst = inst
        self._dilation = dilation
        self.blocks = []
        for i in range(self._number_of_blocks):
            tfun = self._t_func
                
            self.blocks.append(ResnetBlock(self._dim_in if i==0 else self._dim_out, self._dim_out,self._temporal_kernel_size[i], self._stride if i == 0 else 1, tfun, self._dim_inner, self._num_groups, self._stride_1, self._relu_in, dilation=self._dilation))
                
    def call(self, inputs):
        
        x = tensorflow.identity(inputs)
        for i in range(self._number_of_blocks):
            x = self.blocks[i](x)

        return x

    def compute_output_shape(self, input_shape):
        return self.blocks[self._number_of_blocks-1].output_shape