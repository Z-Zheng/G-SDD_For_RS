import os
from dataset.yanshen_reader import read
import cv2
import numpy as np

ROOT = r'D:\rs_cv_yanshen_cup\releasedata'
IMG_DIR = r'images'
LABEL_DIR = r'labelTxt_utf8'

img_filenames = os.listdir(os.path.join(ROOT, IMG_DIR))
label_filenames = os.listdir(os.path.join(ROOT, LABEL_DIR))
img_filenames = sorted(img_filenames)
label_filenames = sorted(label_filenames)


def add_grid(img, shape):
    """add grid to image

    :param img:
    :param shape: tuple,denotes grid shape(h,w)
    :return:
    """
    h, w, c = img.shape
    grid_h = [(i + 1) * int(h / shape[0]) for i in range(shape[0] - 1)]
    grid_w = [(i + 1) * int(w / shape[1]) for i in range(shape[1] - 1)]
    for i in grid_h:
        cv2.line(img, (0, i), (w, i), 3)
    for i in grid_w:
        cv2.line(img, (i, 0), (i, w), 3)


def get_num_of_objects():
    ref_t = {}
    for imgfilename, label_filename in zip(img_filenames, label_filenames):
        abs_imgfn = os.path.join(ROOT, IMG_DIR, imgfilename)
        abs_labelfn = os.path.join(ROOT, LABEL_DIR, label_filename)
        example = read(abs_imgfn, abs_labelfn, read_img=False)
        for bbox in example.bboxes:
            if bbox.category not in ref_t:
                ref_t[bbox.category] = 0
            ref_t[bbox.category] += 1
    print(ref_t)


def show_image_with_objects(idx):
    image_filename = img_filenames[idx]
    label_filename = label_filenames[idx]
    abs_imgfn = os.path.join(ROOT, IMG_DIR, image_filename)
    abs_labelfn = os.path.join(ROOT, LABEL_DIR, label_filename)
    example = read(abs_imgfn, abs_labelfn, read_img=False)
    img = cv2.imread(abs_imgfn)
    img = cv2.resize(img, (800, 800))
    # add grid to image
    add_grid(img, (16, 16))
    for bbox in example.bboxes:
        coord = np.asarray(bbox.coord).astype(np.int32)
        new_coord = coord * 0.2
        new_coord = new_coord.astype(np.int32)
        cv2.rectangle(img, tuple(new_coord[:2]), tuple(new_coord[2:]), 3)
    cv2.imshow("", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    show_image_with_objects(0)
    