from keras.models import Model
from keras import backend as K
from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
import numpy as np

def CNN_autoencoder(img_shape):
    input_img = Input(shape=img_shape)
    # Encoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same',strides=(2,2),name='Conv2D_1')(input_img)
    x = Conv2D(32, (3, 3), activation='relu', padding='same',strides=(2,2),name='Conv2D_2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',strides=(2,2),name='Conv2D_3')(x)
    

    shape_before_flattening = K.int_shape(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = Flatten()(x)
    encoded = Dense(265, activation='relu', name='encoded')(x)

    # Decoder
    x = Dense(np.prod(shape_before_flattening[1:]),
                activation='relu', name='dense')(encoded)
    # Reshape into an image of the same shape as before our last `Flatten` layer
    x = Reshape(shape_before_flattening[1:])(x)

    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same',strides=(2,2),name='Conv2DTranspose_1')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same',strides=(2,2),name='Conv2DTranspose_2')(x)
    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same',strides=(2,2),name='Conv2DTranspose_3')(x)

    decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same',name='decoded')(x)

    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')






# CAE = CNN_autoencoder(img_shape = (112, 112, 3))
pretrain_epochs = 100
batch_size = 40


def pretrain_model(x,autoencoder,save_dir):
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)
    autoencoder.save_weights(save_dir+'/model.h5')

    return autoencoder





