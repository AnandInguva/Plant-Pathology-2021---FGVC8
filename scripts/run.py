import tensorflow as tf
import numpy as np
import argparse
import os

from models.load_model import load as load_model
from scripts.load_data import load_data

AUTOTUNE = tf.data.experimental.AUTOTUNE
num_classes = 12
SCRIPT_DIR = os.path.dirname(__file__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_name', type=str, default='testdelete')
    parser.add_argument('--model_name', type=str, default='mobilenet')
    parser.add_argument('--unfreeze_layers', action='store_true')
    parser.add_argument('--random_weights', action='store_true')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--save_dir_overwrite', type=str, default=None)
    args = parser.parse_args()

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
    model = load_model(model_name=args.model_name,
                       num_classes=num_classes,
                       random_weights=args.random_weights,
                       unfreeze_layers=args.unfreeze_layers,
                       input_img_size=args.image_size)

    print(model.summary())
    ################################


if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(0)
    tf.compat.v1.set_random_seed(0)
    main()
