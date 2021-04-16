import albumentations
import numpy as np
import tensorflow as tf


def strong_aug(p=0.9, img_size=512):
    transform = albumentations.Compose([
        # albumentations.RandomResizedCrop(img_size, CFG.img_size, scale=(0.9, 1), p=1),
        albumentations.HorizontalFlip(p=0.7),
        albumentations.VerticalFlip(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
        albumentations.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7),
        albumentations.CLAHE(clip_limit=(1, 4), p=0.5),
        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.2),
        albumentations.OneOf([
            albumentations.GaussNoise(var_limit=(10, 50)),
            albumentations.GaussianBlur(),
            albumentations.MotionBlur(),
            albumentations.MedianBlur(),
        ], p=0.2),
        #   albumentations.Resize(CFG.img_size, CFG.img_size),
        albumentations.OneOf([
            #  albumentations.JpegCompression(),
            albumentations.Downscale(scale_min=0.1, scale_max=0.15),
        ], p=0.5),
        albumentations.IAAPiecewiseAffine(p=0.2),
        albumentations.IAASharpen(p=0.2),
        # albumentations.Cutout(max_h_size=int(img_size * 0.1), max_w_size=int(img_size * 0.1), num_holes=5, p=0.5),
    ])
    return transform


img_size = 512
# work on the augmentation functions
transforms = strong_aug(img_size=img_size)


def augment_image(image):
    data = {'image': image}
    data_transform = transforms(**data)
    img_transformed = data_transform['image']
    return img_transformed


def tf_augment_image(images):
    images = tf.numpy_function(augment_image, inp=[images], Tout=[images.dtype])
    return images[0]


def augment_batches(images, labels):
    # tf.map_fn : takes batches of images, unroll the batches
    # perfom op on every image in batch and rolls back to batch
    aug_images = tf.map_fn(tf_augment_image, images)
    aug_images = tf.reshape(aug_images, shape=tf.shape(images))
    return aug_images, labels


def augment_in_model(images):
    aug_images = tf.map_fn(tf_augment_image, images)
    aug_images = tf.reshape(aug_images, shape=tf.shape(images))
    return aug_images


class AugmentationLayer(tf.keras.layers.Layer):

    def __init__(self, augmentation_function=augment_in_model):
        super(AugmentationLayer, self).__init__()
        self.augmentation_function = augmentation_function

    def call(self, images, training=False):
        if not training:
            return images
        augmented_images = self.augmentation_function(images)
        return augmented_images

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'augmentation_function': self.augmentation_function
        })
        return config