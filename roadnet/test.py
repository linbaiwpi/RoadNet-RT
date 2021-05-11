import keras as K
import keras.layers as L
import keras.models as M



def ConvBNReLU(x, filter=64, kernel_size=3, strides=1, use_bias=True, name="ConvBNReLU"):
    x = L.Conv2D(filters=filter, kernel_size=kernel_size, strides=(strides, strides),
                 padding="same", use_bias=use_bias, name=name+"_conv")(x)
    x = L.BatchNormalization(name=name+"_bn")(x)
    x = L.ReLU(name=name+"_relu")(x)
    return x


def AttentionRefinement(x, out_channel=64):
    x = ConvBNReLU(x, out_channel, 3, 1, use_bias=False, name="ARM")
    print("==========================================")
    print(x.shape)
    x_sc = L.GlobalAveragePooling2D(data_format='channels_last', name="avg_pool")(x)
    x_sc = L.Reshape((1, 1, x_sc.shape[1]))(x_sc)
    print(x_sc.shape)
    x_sc = L.Conv2D(out_channel, 1, strides=1, padding="same", name="ARM_1x1")(x_sc)
    x_sc = L.Activation('sigmoid')(x_sc)
    x = L.multiply([x_sc, x])
    return x


def FeatureFusion(sp_out, cp_out):
    ffm_cat = L.concatenate([sp_out, cp_out])
    ffm_conv = ConvBNReLU(ffm_cat, 64, 1, 1, use_bias=False, name="ffm_conv1")
    print("==========================================")
    print(ffm_conv.shape)
    # ffm_cam = L.GlobalAveragePooling1D(data_format='channels_last')(ffm_conv)
    ffm_cam = ffm_conv
    ffm_cam = L.Conv2D(filters=64, kernel_size=1, strides=1, padding="same", use_bias=False, name="ffm_conv2")(ffm_cam)
    ffm_cam = L.ReLU()(ffm_cam)
    ffm_cam = L.Conv2D(filters=64, kernel_size=1, strides=1, padding="same", use_bias=False, name="ffm_conv3")(ffm_cam)
    ffm_cam = L.Activation('sigmoid')(ffm_cam)
    ffm_cam = L.multiply([ffm_conv, ffm_cam])
    ffm_cam = L.add([ffm_cam, ffm_conv])
    return ffm_cam


class BiSeNet_lite3():
    def __init__(self, input_shape=(160, 600), num_class=2, activation='sigmoid'):
        self.input_shape = input_shape
        self.num_class = num_class
        self.activation = activation

    def build(self):
        x_in = L.Input(shape=(1, 1, 3))
        # spatial path

        #x_sc = L.GlobalAveragePooling2D(data_format='channels_last', name="avg_pool")(x)
        x_sc = L.Reshape((1, 1, 3))(x_in)
        x_sc = L.Conv2D(6, 1, strides=1, padding="same", name="ARM_1x1")(x_in)
        return M.Model(inputs=x_in, outputs=x_sc)

# based on BiSeNet_lite, replae the AvePooling to GlobalAvgPooling2D


if __name__ == '__main__':
    model = BiSeNet_lite3((1, 1), 1).build()
    model.summary()
