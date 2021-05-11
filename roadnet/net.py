import keras as K
import keras.layers as L
import keras.models as M
import tensorflow as tf

def resnetLayer(x_in, filters, strides, name):
    # main branch
    x = L.Conv2D(filters=filters, kernel_size=3, strides=strides, padding="same", name=name+"_conv1")(x_in)
    x = L.BatchNormalization(name=name+"_bn1")(x)
    x = L.ReLU(name=name+"_relu1")(x)
    x = L.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", name=name+"_conv2")(x)
    x = L.BatchNormalization(name=name+"_bn2")(x)
    # shortcut
    x_sc = L.Conv2D(filters=filters, kernel_size=3, strides=strides, padding="same", name=name+"_conv_sc")(x_in)
    x_sc = L.BatchNormalization(name=name+"_bn_sc")(x_sc)
    # add
    x = L.add([x, x_sc])
    x = L.ReLU(name=name+"_relu2")(x)
    return x


def ConvBNReLU(x, filter=64, kernel_size=3, strides=1, use_bias=True, name="ConvBNReLU"):
    x = L.Conv2D(filters=filter, kernel_size=kernel_size, strides=(strides, strides),
                 padding="same", use_bias=use_bias, name=name+"_conv")(x)
    x = L.BatchNormalization(name=name+"_bn")(x)
    x = L.ReLU(name=name+"_relu")(x)
    return x


def atrousConvBNReLU(x, filter=64, kernel_size=3, strides=1, dilation_rate=2, use_bias=True, name="aConvBNReLU"):
    x = L.Conv2D(filters=filter, kernel_size=kernel_size, strides=(strides, strides), dilation_rate=(dilation_rate, dilation_rate),
                 padding="same", use_bias=use_bias, name=name+"_conv")(x)
    x = L.BatchNormalization(name=name+"_bn")(x)
    x = L.ReLU(name=name+"_relu")(x)
    return x


def AttentionRefinement(x, out_channel=64):
    x = ConvBNReLU(x, out_channel, 3, 1, use_bias=False, name="ARM")
    x_sc = L.GlobalAveragePooling2D(data_format='channels_last', name="avg_pool")(x)
    x_sc = L.Reshape((1, 1, out_channel))(x_sc)
    x_sc = L.Conv2D(out_channel, 1, strides=1, padding="same", name="ARM_1x1")(x_sc)
    x_sc = L.Activation('sigmoid')(x_sc)
    x = L.multiply([x_sc, x])
    return x


def FeatureFusion(sp_out, cp_out):
    ffm_cat = L.concatenate([sp_out, cp_out])
    ffm_conv = ConvBNReLU(ffm_cat, 64, 1, 1, use_bias=False, name="ffm_conv1")
    ffm_cam = L.GlobalAveragePooling2D(data_format='channels_last')(ffm_conv)
    ffm_cam = L.Reshape((1, 1, 64))(ffm_cam)
    ffm_cam = L.Conv2D(filters=64, kernel_size=1, strides=1, padding="same", use_bias=False, name="ffm_conv2")(ffm_cam)
    ffm_cam = L.ReLU()(ffm_cam)
    ffm_cam = L.Conv2D(filters=64, kernel_size=1, strides=1, padding="same", use_bias=False, name="ffm_conv3")(ffm_cam)
    ffm_cam = L.Activation('sigmoid')(ffm_cam)
    ffm_cam = L.multiply([ffm_conv, ffm_cam])
    ffm_cam = L.add([ffm_cam, ffm_conv])
    return ffm_cam


class roadnet_rt():
    def __init__(self, input_shape=(160, 600), num_class=2, activation='sigmoid'):
        self.input_shape = input_shape
        self.num_class = num_class
        self.activation = activation

    def build(self):
        x_in = L.Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        # spatial path
        convBnRelu = ConvBNReLU(x_in, 64, 7, 2, use_bias=False, name="convBnRelu")
        convBnRelu_1 = ConvBNReLU(convBnRelu, 64, 3, 2, use_bias=False, name="convBnRelu1")
        convBnRelu_2 = ConvBNReLU(convBnRelu_1, 64, 3, 2, use_bias=False, name="convBnRelu2")
        sp_out = ConvBNReLU(convBnRelu_2, 128, 1, 1, use_bias=False, name="convBnRelu3")

        # context path
        #x_div_2 = L.Lambda(lambda image: K.backend.resize_images(image, 0.5, 0.5, 'channels_last', 'bilinear'))(x_in)
        # x_div_2 = L.Lambda(lambda image: tf.image.resize_images(image, (int(self.input_shape[0]/2), int(self.input_shape[1]/2))))(x_in)
        x_div_2 = L.Lambda(lambda image: tf.image.resize(image, (int(self.input_shape[0]/2), int(self.input_shape[1]/2))))(x_in)
        cp_x = L.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", use_bias=False, name="cp_conv")(x_div_2)
        cp_x = resnetLayer(cp_x, 64, 2, "backbone_1")
        cp_x = resnetLayer(cp_x, 128, 2, "backbone_2")

        cp_x_1 = atrousConvBNReLU(cp_x, 32, 3, 1, 2, use_bias=False, name="aconv1")
        cp_x_2 = atrousConvBNReLU(cp_x, 32, 3, 1, 4, use_bias=False, name="aconv2")
        cp_x_3 = atrousConvBNReLU(cp_x, 32, 3, 1, 8, use_bias=False, name="aconv3")
        cp_x_4 = atrousConvBNReLU(cp_x, 32, 3, 1, 16, use_bias=False, name="aconv4")
        cp_x = L.concatenate([cp_x_1, cp_x_2, cp_x_3, cp_x_4])

        cp_arm = AttentionRefinement(cp_x, 64)
        cp_out = L.concatenate([cp_arm, cp_x])

        # fusion
        ffm_cam = FeatureFusion(sp_out, cp_out)
        ffm_cam = L.Conv2D(filters=self.num_class, kernel_size=3, strides=1, padding="same", use_bias=False, name="output")(ffm_cam)
        x_out = L.Activation(self.activation)(ffm_cam)
        x_out = L.UpSampling2D(size=8, interpolation='bilinear', name="upsample")(x_out)
        return M.Model(inputs=x_in, outputs=x_out)

# based on BiSeNet_lite, replae the AvePooling to GlobalAvgPooling2D
# move AttentionRefinement to context path
# first conv kernel size = 7

if __name__ == '__main__':
    model = BiSeNet_mod4_base3((280, 960), 1).build()
    model.summary()
