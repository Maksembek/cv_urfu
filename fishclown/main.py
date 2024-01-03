import cv2
import glob
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from utils.compute_iou import compute_ious

def segment_fish(img):
    """
    Method compute masks given image
    Params:    img (np.ndarray): given image in BGR    
    Returns:   mask (np.ndarray): mask, contain bool values
    """

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    # using MORPH_CLOSE
    close_img = cv2.morphologyEx(hsv_image, cv2.MORPH_CLOSE, kernel)

    # lower and upper bounds for the first color filter
    light_orange = [1, 190, 150]
    dark_orange = [30, 255, 255]
    lower_orange = np.array(light_orange, np.uint8)
    upper_orange = np.array(dark_orange, np.uint8)

    # lower and upper bounds for the second color filter
    light_white = [60, 0, 200]
    dark_white = [145, 150, 255]
    lower_white = np.array(light_white, np.uint8)
    upper_white = np.array(dark_white, np.uint8)

    # create two masks using the color ranges
    orange_mask = cv2.inRange(close_img, lower_orange, upper_orange)
    white_mask = cv2.inRange(close_img, lower_white, upper_white)

    # combine two masks using bitwise or operation
    combined_mask = cv2.bitwise_or(orange_mask, white_mask)

    # return np.array(mask)
    return combined_mask


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    args = parser.parse_args()
    stage = 'train' if args.is_train else 'test'

    data_root = osp.join("dataset", stage, "imgs")
    img_paths = glob.glob(osp.join(data_root, "*.jpg"))
    len(img_paths)

    if len(img_paths) == 0:
        print("no img")
    else:
        masks = dict()
        for path in img_paths:
            img = cv2.imread(path)
            mask = segment_fish(img)
            masks[osp.basename(path)] = mask

        print(compute_ious(masks, osp.join("dataset", stage, "masks")))