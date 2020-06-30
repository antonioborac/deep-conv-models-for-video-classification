import tensorflow
from tensorflow.keras import layers

class FuseFastConv(layers.Layer):
    def __init__(self, dim_in, channel_ratio, fusion_kernel, alpha, eps=1e-5, bn_mnt=0.1, normalization_method=tensorflow.keras.layers.BatchNormalization):
        super(FuseFastConv,self).__init__()
        self.dim_in = dim_in
        self.channel_ratio = channel_ratio
        self.dim_out = dim_in*channel_ratio
        self.fusion_kernel = fusion_kernel
        self.alpha = alpha
        self.epsilon = eps
        self.bn_mnt = bn_mnt
        self.normalization_method = normalization_method
        
    def build(self, input_shape):
        self.conv_pad = tensorflow.keras.layers.ZeroPadding3D(padding=[self.fusion_kernel // 2, 0, 0])
       
        self.conv = tensorflow.keras.layers.Conv3D(self.dim_out, kernel_size=[self.fusion_kernel, 1, 1], strides=[self.alpha,1,1], padding='valid',use_bias=False)
        self.bn = self.normalization_method(axis=-1,momentum=self.bn_mnt, epsilon=self.epsilon)
        self.relu = tensorflow.keras.layers.ReLU()
        self.concat = tensorflow.keras.layers.Concatenate(axis=-1)

        super(FuseFastConv,self).build(input_shape)
    def call(self,inputs,training=None):
        slow = tensorflow.identity(inputs[0])
        fast = tensorflow.identity(inputs[1])
        fused = self.conv_pad(fast)
        fused = self.conv(fused)
        fused = self.bn(fused, training=training)
        fused = self.relu(fused)
        fused = self.concat([slow,fused])
        return fused

    def compute_output_shape(self, input_shape):
        return self.concat.output_shape