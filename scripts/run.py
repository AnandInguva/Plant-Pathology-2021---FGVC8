import tensorflow as tf
import numpy as np
import argparse
import os
import tensorflow_addons as tfa

from models.load_model import load as load_model
from scripts.load_data import load_data
from augmentations.augmentations import AugmentationLayer
from utils.utils import *
from utils.callbacks import *

AUTOTUNE = tf.data.experimental.AUTOTUNE
num_classes = 12
SCRIPT_DIR = os.path.dirname(__file__)
tfdata_buffer_size = 1000
seed = 0


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_name', type=str, default='testdelete')
    parser.add_argument('--model_name', type=str, default='mobilenet')
    parser.add_argument('--unfreeze_layers', action='store_true')
    parser.add_argument('--random_weights', action='store_true')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--save_dir_overwrite', type=str, default=None)
    parser.add_argument('--do_aug', action='store_true')
    parser.add_argument('--balance_batches', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loss', type=str, default='binary_cross_entropy')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=200)
    args = parser.parse_args()

    image_size = args.image_size
    batch_size = args.batch_size
    ################################
    # load data
    if args.save_dir_overwrite is None:
        save_dir = os.path.abspath(os.path.join(os.path.join(SCRIPT_DIR, '../exps'), args.exp_name))  # save locally
    else:
        # save to the output mount location
        # when running a gcp experiment
        # args.save_dir_overwrite = '/root/anand/' -> run_gcp.py
        save_dir_root = os.path.expanduser(args.save_dir_overwrite)
        save_dir = os.path.join(save_dir_root, args.exp_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/model_weights')
        os.makedirs(save_dir + '/metrics')
        os.makedirs(save_dir + '/visuals')

    f = open(os.path.join(save_dir, 'args.txt'), 'w')
    f.write('args = ' + repr(args) + '\n')
    f.close()

    data_filepath = os.path.abspath(os.path.join(SCRIPT_DIR, '../DATA'))

    ################################
    train_filenames, val_filenames, train_multi_label_labels, val_multi_label_labels, train_multi_class_labels, val_multi_class_labels, train_labels_name, val_labels_name, num_classes, unique_labels, class_mapping, label_to_cat_label_map = load_data(
        data_filepath)

    ################################
    # load model
    if args.do_aug:
        augmentation_layer = AugmentationLayer()
    else:
        augmentation_layer = None
    model = load_model(model_name=args.model_name,
                       num_classes=num_classes,
                       random_weights=args.random_weights,
                       unfreeze_layers=args.unfreeze_layers,
                       input_img_size=args.image_size,
                       augmentation_layer=augmentation_layer)

    print(model.summary())
    ################################
    # create train and val dataloader
    train_ds = create_tf_map_dataset(image_size, train_filenames, train_multi_label_labels, train_labels_name,
                                     normalize_fn=get_image_dontnormalize)
    val_ds = create_tf_map_dataset(image_size, val_filenames, val_multi_label_labels, val_labels_name,
                                   normalize_fn=get_image_dontnormalize)
    ################################
    # create sampled dataset if needed
    if args.balance_batches:
        desired_fraction = np.array([1 / len(class_mapping)] * len(class_mapping))
        dfs = create_sampled_dataset(train_ds, unique_labels)
        train_data = tf.data.experimental.sample_from_datasets(dfs, desired_fraction)
        steps_per_epoch = max(1, 2 * len(train_filenames) // batch_size)
    else:
        steps_per_epoch = max(1, len(train_filenames) // batch_size)

    ###############################
    # Batch train and val_data
    train_data = train_ds.shuffle(buffer_size=tfdata_buffer_size, reshuffle_each_iteration=True,
                                  seed=seed)
    train_data = train_data.batch(batch_size).prefetch(AUTOTUNE)
    val_data = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    train_data = train_data.map(lambda images, categ_labels, labels, filename: (images, categ_labels),
                                num_parallel_calls=AUTOTUNE)
    val_data = val_data.map(lambda images, categ_labels, labels, filename: (images, categ_labels),
                            num_parallel_calls=AUTOTUNE)
    ###############################
    # define callbacks
    save_best_model_cbk_on_val_loss = saveModelOnMetric(save_dir,
                                                        metric='val_loss',
                                                        mode='min',
                                                        save_weights_only=False)
    save_best_model_cbk_on_val_f1 = saveModelOnMetric(save_dir,
                                                      metric='val_F1',
                                                      mode='max',
                                                      save_weights_only=False)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(save_dir, 'metrics.csv'), append=True)

    callbacks = [save_best_model_cbk_on_val_f1, save_best_model_cbk_on_val_loss, csv_logger]
    ###############################
    # Define loss function
    if args.loss == 'binary_cross_entropy':
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    elif args.loss == 'categorical_cross_entropy':
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    else:
        raise NotImplementedError
    ###############################
    # Define optimizer
    learning_rate = args.learning_rate
    if args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise NotImplementedError
    ###############################
    # Define metrics
    metrics = ['Accuracy', tfa.metrics.F1Score(num_classes=len(unique_labels),
                                               average='macro',
                                               name='F1')]
    ###############################
    # Compile model
    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=metrics)
    ###############################
    # train model
    history = model.fit(train_data.repeat(),
                        epochs=args.num_epochs,
                        steps_per_epoch=steps_per_epoch,
                        initial_epoch=0,
                        validation_data=val_data,
                        callbacks=callbacks)


if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(0)
    tf.compat.v1.set_random_seed(0)
    main()
