import sys

from keras import applications
from keras.layers import Conv2D, LeakyReLU, Concatenate, SeparableConv2D, BatchNormalization
from keras.models import Model, load_model

from layers import BilinearUpSampling2D
from loss import depth_loss_function


def create_model(existing='', is_twohundred=False, is_halffeatures=True):
    if len(existing) == 0:
        print('Loading base model (DenseNet)..')

        # Applying pre-trained DenseNet-169 asEncoder Layers
        base_model = applications.DenseNet169(input_shape=(None, None, 3), include_top=False)

        print('Base model loaded.')

        # Starting point for decoder
        base_model_output_shape = base_model.layers[-1].output.shape

        # Setting up Layer freezing
        for layer in base_model.layers: layer.trainable = True

        # Starting number of decoder filters
        if is_halffeatures:
            decode_filters = int(int(base_model_output_shape[-1]) / 2)
        else:
            decode_filters = int(base_model_output_shape[-1])

        # Define upsampling layer
        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name + '_upsampling2d')(tensor)
            up_i = Concatenate(name=name + '_concat')(
                [up_i, base_model.get_layer(concat_with).output])  # Skip Connection
            up_i = SeparableConv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convA')(
                up_i)  # Separable Convolution
            up_i = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                      beta_initializer="zeros", gamma_initializer="ones")(
                up_i)  # Batch Normalization to Avoid Overfitting
            up_i = LeakyReLU(alpha=0.2)(up_i)  # Leaky version of a Rectified Linear Unit
            up_i = SeparableConv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convB')(
                up_i)  # Separable Convolution
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape,
                         name='conv2')(base_model.output)

        decoder = upproject(decoder, int(decode_filters / 2), 'up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters / 4), 'up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters / 8), 'up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters / 16), 'up4', concat_with='conv1/relu')
        if False:
            decoder = upproject(decoder, int(decode_filters / 32), 'up5', concat_with='input_1')

        # Extract depths in the Final Layer
        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

        # Create the Model
        model = Model(inputs=base_model.input, outputs=conv3)
    else:
        # Load existing model from filesystem
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')

        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects)
        print('\nExisting model loaded.\n')

    print('Model created.')

    return model
