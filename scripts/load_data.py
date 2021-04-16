import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf

num_folds = 5
seed = 42


def load_data(path, validation_fold=4):
    labels_path = os.path.join(path, 'train.csv')
    images_path = os.path.join(path, 'train_downsized')

    df = pd.read_csv(labels_path)

    unique_labels = sorted(df['labels'].unique())
    num_classes = len(unique_labels)

    print("Total classes : ", num_classes)
    print("Total Samples : ", len(df))

    class_labels = df['labels'].tolist().copy()
    df['labels'] = df['labels'].apply(lambda x: x.split(' '))
    df['underlying_labels'] = class_labels

    ##########################################
    # removing duplicates

    duplicates_path = os.path.join(path, 'duplicates.pickle')
    duplicates = pickle.load(open(duplicates_path, 'rb'))

    duplicates_to_remove = []
    for key, val in duplicates.copy().items():
        if not val:
            del duplicates[key]
        else:
            duplicates_to_remove.extend(val)
    duplicates_to_remove = set(duplicates_to_remove)

    duplicates_mask = df['image'].apply(lambda x: x not in duplicates_to_remove)
    print('Length of DF before removing duplicates : {}'.format(len(df)))
    df = df[duplicates_mask]
    print('Length of DF after removing duplicates : {}'.format(len(df)))
    ############################################
    # creating multilabels for each class
    for i, class_name in enumerate(unique_labels):
        df[class_name] = df['labels'].apply(lambda x: int(class_name in x))

    df['fold'] = 0

    mskf = StratifiedShuffleSplit(n_splits=num_folds, random_state=seed, test_size=0.2)
    for i, (train_index, test_index) in enumerate(mskf.split(df['image'], df['labels'])):
        df.iloc[test_index, -1] = i

    df['fold'] = df['fold'].astype('int')
    mask = df['fold'] == validation_fold
    df['is_valid'] = mask

    train_df = df[df['is_valid'] == False]
    val_df = df[df['is_valid'] == True]

    class_mapping = {}

    for i in range(len(unique_labels)):
        class_mapping[unique_labels[i]] = i

    label_to_cat_label_map = {}
    for key, value in class_mapping.items():
        label_to_cat_label_map[value] = key

    train_df['encoded_labels'] = train_df['underlying_labels'].apply(lambda x: class_mapping[x])
    val_df['encoded_labels'] = val_df['underlying_labels'].apply(lambda x: class_mapping[x])

    #########################################
    # load train and val filenames, labels
    ########################################
    train_filenames = train_df['image'].tolist()
    train_filenames = [os.path.join(images_path, fname) for fname in train_filenames]
    val_filenames = val_df['image'].tolist()
    val_filenames = [os.path.join(images_path, fname) for fname in val_filenames]

    train_multi_label_labels = train_df[unique_labels].to_numpy()
    val_multi_label_labels = val_df[unique_labels].to_numpy()

    train_multi_class_labels = tf.keras.utils.to_categorical(train_df['encoded_labels'].tolist())
    val_multi_class_labels = tf.keras.utils.to_categorical(val_df['encoded_labels'].tolist())

    train_labels_name = train_df['underlying_labels'].tolist()
    val_labels_name = val_df['underlying_labels'].tolist()

    print(df.head(15))
    return train_filenames, val_filenames, train_multi_label_labels, val_multi_label_labels, train_multi_class_labels, val_multi_class_labels, train_labels_name, val_labels_name, num_classes, unique_labels, class_mapping, label_to_cat_label_map
