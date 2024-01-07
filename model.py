import tensorflow as tf
from tensorflow import keras


def model_def():


    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B1(


    weights = 'imagenet',

    include_preprocessing = False,


    include_top=False
    )

    avg = keras.layers.GlobalMaxPool2D()(base_model.output)

    output = keras.layers.Dense(7,activation='softmax')(avg)

    model = keras.Model(inputs = base_model.input,outputs = output)

    for layer in base_model.layers[:-3]:
        layer.trainable = False

    #model.build()
    model.summary()
    return(model)
