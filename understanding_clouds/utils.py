import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from understanding_clouds.constants import NO_MASK_PROVIDED, BACKGROUND_CLASSNAME

def preproces_dataframe_single_mask(df):
    df['filename'] = df['Image_Label'].apply(lambda x: x.split('_')[0])
    df['mask_type'] = df['Image_Label'].apply(lambda x: x.split('_')[1])
    return df

def preproces_dataframe_all_masks(df):
    df = preproces_dataframe_single_mask(df)
    orig_index = df.filename.drop_duplicates().tolist()
    df['EncodedPixels'].fillna(NO_MASK_PROVIDED, inplace=True)
    df_mask = df['EncodedPixels']==NO_MASK_PROVIDED
    df.loc[df_mask,'mask_type']=BACKGROUND_CLASSNAME
    df = df.drop('Image_Label',axis=1)
    df = df.groupby('filename').transform(lambda x: ','.join(x)).drop_duplicates()
    df['filename']=orig_index
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
        return np.zeros((height, width))
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

def get_all_masks_and_img(df, index, images_dirpath):
    img_path = df.iloc[index]['filename']
    img = cv2.imread(os.path.join(images_dirpath, img_path))
    w, h = img.shape[:2]
    rle_masks = df.iloc[index]['EncodedPixels']
    rle_masks = rle_masks.split(',')
    masks = [rle_to_mask(rle_mask, h, w) for rle_mask in rle_masks]
    labels = df.iloc[index]['mask_type']
    labels = labels.split(',')
    return masks, img, labels


def get_mask_and_img(df, index, images_dirpath):
    img_path = df.loc[index, 'filename']
    img = cv2.imread(os.path.join(images_dirpath, img_path))
    w, h = img.shape[:2]
    mask = rle_to_mask(df.loc[index, 'EncodedPixels'], h, w)
    return mask, img

def show_masks_and_img(masks, img, labels):
    fig, axs = plt.subplots(len(masks)+1, figsize=(20,60))
    for i, (mask, label) in enumerate(zip(masks,labels)):
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
