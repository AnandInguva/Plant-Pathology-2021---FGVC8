import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_image_dontnormalize(filename, categorical_label, image_size, class_name, preprocess_method=None):
    # decode_jpeg
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.cast(tf.image.resize(image_decoded, (image_size, image_size)),
                            tf.uint8)
    return image_resized, categorical_label, filename, class_name


def create_tf_dataset(image_size,
                      filenames,
                      categ_labels,
                      class_name,
                      normalize_fn=None,
                      preprocess_type='none'):
    image_size_tiled = tf.constant([image_size] * len(filenames))
    preprocess_type_tiled = tf.constant([preprocess_type] * len(filenames))
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(filenames), tf.constant(categ_labels), image_size_tiled,
         tf.constant(class_name),
         preprocess_type_tiled)) \
        .map(normalize_fn, num_parallel_calls=AUTOTUNE)
    return ds


def create_tf_map_dataset(image_size, filenames, categ_labels, class_name, normalize_fn, preprocess_type='none'):
    '''Uses the given filenames and labels to create tf datasets.'''
    train_ds = create_tf_dataset(image_size, filenames, categ_labels, class_name, normalize_fn)
    return train_ds


def create_sampled_dataset(train_ds, unique_labels):
    df_s = []
    for value in unique_labels:
        temp_ds = train_ds.filter(lambda image, categ_label, filename, class_name : class_name == value)
        df_s.append(temp_ds)
    return df_s