import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from understanding_clouds.constants import NO_MASK_PROVIDED, BACKGROUND_CLASSNAME


def collate_fn(batch):
    # unpacking a training batch for mask_rcnn
    return tuple(zip(*batch))


def scale_img(img, scale_factor, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    new_shape = w // scale_factor, h // scale_factor
    img_scaled = cv2.resize(img, new_shape, interpolation=interpolation)
    return img_scaled


def preproces_dataframe_single_mask(df):
    df['filename'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    df['mask_type'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
    return df


def preproces_dataframe_all_masks(df):
    df = preproces_dataframe_single_mask(df)
    orig_index = df.filename.drop_duplicates().tolist()
    df['EncodedPixels'].fillna(NO_MASK_PROVIDED, inplace=True)
    df_mask = df['EncodedPixels'] == NO_MASK_PROVIDED
    df.loc[df_mask, 'mask_type'] = BACKGROUND_CLASSNAME
    df = df.drop('Image_Label', axis=1)
    df = df.groupby('filename').transform(
        lambda x: ','.join(x)).drop_duplicates()
    df['filename'] = orig_index
    return df


def rle_to_mask(rle_string, width, height):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters:
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask

    Returns:
    numpy.array: numpy array of the mask
    '''
    rows, cols = height, width

    if not isinstance(rle_string, str) or rle_string == NO_MASK_PROVIDED:
        return np.zeros((height, width), np.uint8)
    else:
        rle_numbers = [int(num_string)
                       for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1, 2)
        img = np.zeros(rows * cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index + length] = 255
        img = img.reshape(cols, rows).T

        return img


def get_all_masks_and_img(df, index, images_dirpath, scale_factor=4, interpolation=cv2.INTER_AREA):
    img_path = df.iloc[index]['filename']
    img = cv2.imread(os.path.join(images_dirpath, img_path))
    w, h = img.shape[:2]
    rle_masks = df.iloc[index]['EncodedPixels']
    rle_masks = rle_masks.split(',')
    masks = [rle_to_mask(rle_mask, h, w) for rle_mask in rle_masks]
    if scale_factor:
        img = scale_img(img, scale_factor, interpolation)
        masks = [scale_img(mask, scale_factor, interpolation)
                 for mask in masks]
    labels = df.iloc[index]['mask_type']
    labels = labels.split(',')
    return masks, img, labels


def get_mask_and_img(df, index, images_dirpath, scale_factor=4, interpolation=cv2.INTER_AREA):
    img_path = df.loc[index, 'filename']
    img = cv2.imread(os.path.join(images_dirpath, img_path))
    w, h = img.shape[:2]
    mask = rle_to_mask(df.loc[index, 'EncodedPixels'], h, w)
    if scale_factor:
        img = scale_img(img, scale_factor, interpolation)
        mask = scale_img(mask, scale_factor, interpolation)
    return mask, img


def show_masks_and_img(masks, img, labels):
    fig, axs = plt.subplots(len(masks) + 1, figsize=(20, 60))
    for i, (mask, label) in enumerate(zip(masks, labels)):
        axs[i].imshow(mask)
        axs[i].title.set_text(f'Mask type is: {label}')
    axs[-1].imshow(img)
    plt.show()


def show_mask(df, index, images_dirpath):
    mask_name = df.loc[index, 'mask_type']
    mask, img = get_mask_and_img(df, index, images_dirpath)
    fig, axs = plt.subplots(2, figsize=(15, 15))
    axs[0].imshow(mask)
    axs[1].imshow(img)
    plt.title(f'Mask type is: {mask_name}')
    plt.show()


def plot_losses(data_train, data_valid):
    fig, axs = plt.subplots(figsize=(30,20), ncols=3, nrows=2, squeeze=False)
    axs = axs.reshape(6)
    for i, loss_type in enumerate(data_train[0].keys()):
        axs[i].plot([l[loss_type] for l in data_train], label='TRAIN')
        axs[i].plot([l[loss_type] for l in data_valid], label='VALID')
        axs[i].legend()
        axs[i].set_title(loss_type)
    plt.show()

def get_losses(data, phase):
    return [losses for epoch_value in data.values() for losses in epoch_value[phase]['per_batch_losses'].values()]
