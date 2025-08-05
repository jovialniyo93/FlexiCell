from os import listdir
from os.path import splitext
from glob import glob
import cv2
import numpy as np

# Contrast augmentation function
def contrast_augmentation(image, contrast_value):
    # Adjust the contrast by multiplying the image with a contrast value
    image = image.astype(np.float32)  # Convert to float to avoid overflow
    image_contrasted = np.clip(image * contrast_value, 0, 255)  # Clip to stay in the valid range
    return image_contrasted.astype(np.uint8)  # Convert back to uint8 for saving

# Main augmentation function for contrast only
def augment_contrast_and_replace(imgs_dir, mask_dir, sequence, start_num):
    image_file = glob(imgs_dir + sequence + '.*')[0]  # Get image path
    mask_file = glob(mask_dir + sequence.replace('t', 'man_seg') + '.*')[0]  # Get mask path

    # Load the image and mask
    image = cv2.imread(image_file, -1)
    mask = cv2.imread(mask_file, -1)

    # Contrast values from 0.8 to 1.5
    contrast_list = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    # Replace original image with 0.8 contrast version
    image_augmented = contrast_augmentation(image, contrast_list[0])
    cv2.imwrite(image_file, image_augmented)  # Overwrite the original image with the 0.8 contrast version

    # Save augmented versions for remaining contrast values (0.9 to 1.5)
    for i, contrast_value in enumerate(contrast_list[1:], start=1):
        # Apply contrast augmentation for the rest of the contrast values
        image_augmented = contrast_augmentation(image, contrast_value)

        # Sequential file name generation
        image_num_str = str(start_num + i - 1).zfill(6)
        image_file_aug = imgs_dir + image_num_str + '.tif'
        mask_file_aug = mask_dir + image_num_str + '.tif'

        # Save augmented images
        cv2.imwrite(image_file_aug, image_augmented)
        cv2.imwrite(mask_file_aug, mask)  # Mask remains the same

    return start_num + len(contrast_list) - 1  # Update the starting number for the next set of images

if __name__ == "__main__":
    imgs_dir = 'data/imgs/'
    mask_dir = 'data/mask/'

    # Get list of image IDs without file extensions
    ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
    print(f"Found image sequences: {ids}")

    start_num = len(ids)  # Initialize the starting number for sequential file naming

    # Sequentially process each image for contrast augmentation and replacement
    for sequence in ids:
        start_num = augment_contrast_and_replace(imgs_dir, mask_dir, sequence, start_num)

    print("Contrast augmentation has finished!")
