import yanshen_reader
import label_map
import os
import random
import string
from skimage import transform
import numpy as np


def parse_cut_file(cut_filename):
    with open(cut_filename, 'r') as f:
        all = f.read()
    all = all.split('\n')
    all = list(filter(lambda e: e != '', all))
    for line in all:
        datas = line.split('\t')
        _, xmin, ymin, xmax, ymax = tuple(filter(lambda e: e != '', datas))
        yield int(xmin), int(ymin), int(xmax), int(ymax)


def is_inside(bound, box):
    xmin, ymin, xmax, ymax = bound
    bxmin, bymin, bxmax, bymax = box
    return xmin < bxmax and ymin < bymin and bxmax < xmax and bymax < ymax


def search_inside_box(bound, boxes):
    return [box for box in boxes if is_inside(bound, box.coord)]


def select_channel(shape):
    channel = None
    # single channel
    if len(shape) == 2:
        channel = 1
    # three channel
    if len(shape) == 3:
        channel = 3
    return channel


def cut_image(bound, src_image, channel):
    xmin, ymin, xmax, ymax = bound
    if channel == 1:
        return src_image[ymin:ymax, xmin:xmax]
    else:
        return src_image[ymin:ymax, xmin:xmax, :]


def _rotate_clockwise(angle, src_image_shape, boxes):
    nboxes = []
    h, w = (0, 0)
    if len(src_image_shape) == 2:
        h, w, c = src_image_shape
    if len(src_image_shape) == 3:
        h, w, c = src_image_shape
    angle_map = {90: lambda xmin, ymin, xmax, ymax: [h - ymax, xmin, h - ymin, w - xmin],
                 180: lambda xmin, ymin, xmax, ymax: [w - xmax, h - ymax, w - xmin, h - ymin],
                 270: lambda xmin, ymin, xmax, ymax: [ymin, w - xmax, ymax, w - xmin],
                 }
    for box in boxes:
        xmin, ymin, xmax, ymax = box.coord
        nbox = yanshen_reader.BBox(angle_map[angle](xmin, ymin, xmax, ymax), box.category, box.blurred)
        nboxes.append(nbox)
    return nboxes


def rotate_four_dir(example):
    angles = [90, 180, 270]
    for angle in angles:
        nexample = yanshen_reader.Example()
        nexample.image = transform.rotate(example.image, angle)
        nexample.bboxes = _rotate_clockwise(angle, example.image.shape, example.bboxes)
        yield nexample


def shift(example, number):
    # mininum bounding rectangle
    mxmin, mymin, mxmax, mymax = (1 << 21, 1 << 21, 0, 0)
    for box in example.bboxes:
        xmin, ymin, xmax, ymax = box.coord
        mxmin = min(mxmin, xmin)
        mymin = min(mymin, ymin)
        mxmax = max(mxmax, xmax)
        mymax = max(mymax, ymax)
    h, w = (0, 0)
    if len(example.image.shape) == 2:
        h, w = example.image.shape
    if len(example.image.shape) == 3:
        h, w, c = example.image.shape
    # left top right down
    padding = (mxmin, mymin, w - mxmax, h - mymax)
    assert mxmin > 0 and mymin > 0 and w - mxmax > 0 and h - mymax > 0, \
        "padding must be greater than 0"
    channel = select_channel(example.image.shape)
    for i in range(number):
        nexample = yanshen_reader.Example()
        flag = np.random.rand()
        if flag >= 0.5:
            dleft = np.random.randint(1, padding[0], 1)
            dtop = np.random.randint(1, padding[1], 1)
            nexample.image = cut_image((dleft, dtop, w, h), example.image, channel)
            for box in example.bboxes:
                xmin, ymin, xmax, ymax = box.coord
                nbox = yanshen_reader.BBox([xmin + dleft, ymin + dtop, xmax, ymax], box.category, box.blurred)
                nexample.bboxes.append(nbox)
            yield nexample
        else:
            dright = np.random.randint(1, padding[2], 1)
            ddown = np.random.randint(1, padding[3], 1)
            nexample.image = cut_image((0, 0, w - dright, h - ddown), example.image, channel)
            for box in example.bboxes:
                xmin, ymin, xmax, ymax = box.coord
                nbox = yanshen_reader.BBox([xmin, ymin, xmax, ymax], box.category, box.blurred)
                nexample.bboxes.append(nbox)
            yield nexample


def cut(image_filename, label_filename, cut_filename, save_dir):
    example = yanshen_reader.read(image_filename, label_filename, read_img=True)
    # get random prefix
    prefix = ''.join(random.choice(string.ascii_letters) for x in range(7))
    bounds = parse_cut_file(cut_filename)

    src_img = example.image
    # get channel
    channel = select_channel(src_img.shape)

    for id, bound in enumerate(bounds):
        example = yanshen_reader.Example()
        # cut image
        new_image = cut_image(bound, src_img, channel)
        example.image = new_image
        # get inside boxes
        inside_box = search_inside_box(bound, example.bboxes)
        example.bboxes = inside_box

        img_filename = "{0}_{1}".format(prefix, id)
        subfilename = "{0}_{1}.txt".format(prefix, id)
        # save
        example.write_to_file(image_filename=img_filename,
                              image_format='.png',
                              boxes_save_path=os.path.join(save_dir, subfilename))

        # log
        print("[{2}]:{0} has been saved in {1}\n".format(img_filename, save_dir, id))
        print("[{2}]:{0} has been saved in {1}\n".format(subfilename, save_dir, id))
        # rotate
        examples = rotate_four_dir(example)
        for i, e in enumerate(examples):
            img_filename = "{0}_{1}r{2}".format(prefix, id, i)
            subfilename = "{0}_{1}r{2}.txt".format(prefix, id, i)
            e.write_to_file(image_filename=img_filename,
                            image_format='.png',
                            boxes_save_path=os.path.join(save_dir, subfilename))
            print("[{2}]:{0} has been saved in {1}\n".format(img_filename, save_dir, id))
            print("[{2}]:{0} has been saved in {1}\n".format(subfilename, save_dir, id))
        # shift
        examples = shift(example, number=20)
        for i, e in enumerate(examples):
            img_filename = "{0}_{1}r{2}".format(prefix, id, i)
            subfilename = "{0}_{1}r{2}.txt".format(prefix, id, i)
            e.write_to_file(image_filename=img_filename,
                            image_format='.png',
                            boxes_save_path=os.path.join(save_dir, subfilename))
            print("[{2}]:{0} has been saved in {1}\n".format(img_filename, save_dir, id))
            print("[{2}]:{0} has been saved in {1}\n".format(subfilename, save_dir, id))
