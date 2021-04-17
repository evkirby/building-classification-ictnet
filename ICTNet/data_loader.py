import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

batch_size=5

# Put batch_size=725 to iterate by city

data_path = './AerialImageDataset'
labels_folder = 'gt'
images_folder = 'images'
roof_folder = 'roof'

def get_data_loader(data_path = './AerialImageDataset',
                    labels_folder = 'gt',
                    images_folder = 'images',
                    roof_folder = 'roof',
                    batch_size=5):

    train_images_path = os.path.join(data_path, 'train', images_folder)
    train_labels_path = os.path.join(data_path, 'train', labels_folder)

    valid_images_path = train_images_path.replace('train', 'valid')
    valid_labels_path = train_labels_path.replace('train', 'valid')

    train_image_count = len(os.listdir(os.path.join(train_images_path, roof_folder)))
    train_label_count = len(os.listdir(os.path.join(train_labels_path, roof_folder)))

    valid_image_count = len(os.listdir(os.path.join(valid_images_path, roof_folder)))
    valid_label_count = len(os.listdir(os.path.join(valid_labels_path, roof_folder)))

    steps_per_epoch = train_image_count // batch_size
    validation_steps = valid_image_count // batch_size

    print(f'Image count :\t{train_image_count}\nLabel count :\t{2}\n')

    image_files = os.listdir(os.path.join(train_images_path, roof_folder))
    img_height, img_width, _ = cv2.imread(os.path.join(train_images_path, roof_folder, image_files[0])).shape # cv2.COLOR_BGR2RGB

    print(f'image height :\t{img_height} pixels')
    print(f'image width  :\t{img_width} pixels')


    # TODO : Here we can do preprocessing while loading the images
    def image_preprocessing(image):
        return image

    def mask_preprocessing(mask):
        # return mask.reshape((mask.shape[0], mask.shape[1], mask.shape[2], 1)) # exemple
        return mask

    def combine_generator(gen1, gen2):
        while True:
            yield(next(gen1), next(gen2))


    image_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function = image_preprocessing
    )

    mask_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function = mask_preprocessing
    )

    train_image_generator = image_datagen.flow_from_directory(
        train_images_path,
        batch_size = batch_size,
        shuffle=False,
        target_size=(1024, 1024),
        class_mode = None,
        seed=42,
    )

    train_mask_generator = mask_datagen.flow_from_directory(
        train_labels_path,
        batch_size = batch_size,
        shuffle=False,
        target_size=(1024, 1024),
        class_mode = None,
        seed=42,
    )

    valid_image_generator = image_datagen.flow_from_directory(
        valid_images_path,
        batch_size = batch_size,
        shuffle=False,
        target_size=(1024, 1024),
        class_mode = None,
        seed=42,
    )

    valid_mask_generator = mask_datagen.flow_from_directory(
        valid_labels_path,
        batch_size = batch_size,
        shuffle=False,
        target_size=(1024, 1024),
        class_mode = None,
        seed=42,
    )

    train_generator = zip(train_image_generator, train_mask_generator)
    valid_generator = zip(valid_image_generator, valid_mask_generator)

    x_train, y_train = next(train_generator)
    print(np.shape(x_train[0]))
    print(np.shape(y_train[0]))

    return train_generator, valid_generator, steps_per_epoch, validation_steps

