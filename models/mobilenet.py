import tensorflow as tf


def load_model(input_image_size, num_classes, random_weights=False, unfreeze_layers=False,
               augmentation_layer=None):
    if random_weights:
        weights = None
    else:
        weights = 'imagenet'
    base_model = tf.keras.applications.MobileNet(input_shape=(input_image_size, input_image_size, 3),
                                                 include_top=False,
                                                 weights=weights)
    if unfreeze_layers:
        base_model.trainable = True

    output_layer = tf.keras.layers.Dense(num_classes, 'sigmoid')
    preprocessor = tf.keras.applications.mobilenet_v2.preprocess_input

    i = tf.keras.layers.Input([None, None, 3], dtype=tf.uint8)
    x = tf.cast(i, tf.float32)
    # adding augmentation layer
    if augmentation_layer:
        x = augmentation_layer(x)
    x = preprocessor(x)
    x = base_model(x)
    x = tf.keras.layers.Flatten()(x)
    x = output_layer(x)
    model = tf.keras.Model(inputs=[i], outputs=[x])
    return model


