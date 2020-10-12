inputShape = (512, 512, 4)
outputShape = 28

def ModelMaker(inputShape, outputShape):
    
    model = Sequential()

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', bias_initializer='zeros',
                     kernel_regularizer='l2', padding='valid', data_format="channels_last", input_shape=inputShape))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer='l2', padding='same'))
    model.add(PReLU(alpha_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.5, noise_shape=None, seed=None))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer='l2', padding='same'))
    model.add(PReLU(alpha_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(GaussianNoise(0.1))
    
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer='l2', padding='same'))
    model.add(PReLU(alpha_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.5, noise_shape=None, seed=None))

    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer='l2', padding='same'))
    model.add(PReLU(alpha_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(GaussianNoise(0.1))
    
    model.add(DepthwiseConv2D((3, 3), strides=(1, 1), kernel_initializer='glorot_uniform', kernel_regularizer='l2', padding='same'))
    model.add(PReLU(alpha_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.5, noise_shape=None, seed=None))

    model.add(Flatten())

    model.add(Dense(512, kernel_initializer='glorot_uniform', kernel_regularizer='l2', activity_regularizer='l2'))
    model.add(PReLU(alpha_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dropout(0.5, noise_shape=None, seed=None))
    
    model.add(Dense(512, kernel_initializer='glorot_uniform', kernel_regularizer='l2', activity_regularizer='l2'))
    model.add(PReLU(alpha_initializer=VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dropout(0.5, noise_shape=None, seed=None))

    model.add(Dense(outputShape))
    model.add(Activation('softmax'))
    
    return model

#Defining the Convolutional Neural Network

model = ModelMaker(inputShape, outputShape)

print("The Model has been created.")


def ModelMaker(INPUT_SHAPE, OUTPUT_SHAPE):
    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='valid', data_format="channels_last", input_shape=INPUT_SHAPE))

    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.5))
    
    model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest"))
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    
    model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest"))
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    
    model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest"))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    
    model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest"))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('softmax'))
    
    return model
