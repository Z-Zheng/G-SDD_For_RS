import yanshen_reader
import label_map
import os
import random
import string
from skimage.io import imsave
import numpy as np


def parse_cut_file(cut_filename):
    with open(cut_filename, 'r') as f:
        all = f.read()
    all = all.split('\n')
    all = list(filter(lambda e: e != '', all))
    for line in all:
        datas = line.split('\t')
        _, xmin, ymin, xmax, ymax = tuple(filter(lambda e: e != '', datas))
        yield xmin, ymin, xmax, ymax


def is_inside(bound, box):
    xmin, ymin, xmax, ymax = bound
    bxmin, bymin, bxmax, bymax = box
    return xmin < bxmax and ymin < bymin and bxmax < xmax and bymax < ymax


def search_inside_box(bound, boxes):
    return [box for box in boxes if is_inside(bound, box.coord)]


def write_boxes_to_file(boxes, save_path):
    format = '{0} {1} {2} {3} {4} {5} {6} {7} {8}\n'
    with open(save_path, 'w') as f:
        for box in boxes:
            xmin, ymin, xmax, ymax = box.coord
            category_str = label_map.label_map_str[box.category]
            record = format.format(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, category_str)
            f.write(record)


def write_image_to_file(image, filename, img_format='.png'):
    imsave("{0}.{1}".format(filename, img_format), image)


def select_channel(shape):
    # single channel
    if len(shape) == 2:
        channel = 1
    # three channel
    if len(shape) == 3:
        channel = 3
    return channel


def cut_image(bound, src_image, channels):
    xmin, ymin, xmax, ymax = bound
    if channels == 1:
        return src_image[ymin:ymax, xmin:xmax]
    else:
        return src_image[ymin:ymax, xmin:xmax, :]


def cut(image_filename, label_filename, cut_filename, save_dir):
    example = yanshen_reader.read(image_filename, label_filename, read_img=True)
    filename = ''.join(random.choice(string.ascii_letters) for x in range(7))
    bounds = parse_cut_file(cut_filename)

    src_img = example.image
    # get channel
    channel = select_channel(src_img.shape)

    for id, bound in enumerate(bounds):
        new_image = cut_image(bound, src_img, channel)
        # write image saved by png to disk
        write_image_to_file(new_image, "{0}_{1}".format(filename, id), '.png')
        subfilename = "{0}_{1}.txt".format(filename, id)
        inside_box = search_inside_box(bound, example.boxes)
        # write boxes saved by txt to disk
        write_boxes_to_file(inside_box, os.path.join(save_dir, subfilename))
        print("[{2}]:{0} has been saved in {1}\n".format(subfilename, save_dir, id))

