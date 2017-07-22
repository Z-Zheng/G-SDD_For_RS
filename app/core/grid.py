"""split remote sensing image into many patch fixed shape by grid

"""
import numpy as np
from skimage import io as imageio
from dataset.yanshen_reader import Example


def fake_process(image):
    return image


def is_inside(min_iou, bound, box):
    bxmin, bymin, bxmax, bymax = bound
    xmin, ymin, xmax, ymax = box.coord
    if bxmin < xmin and bymin < ymin and xmax < bxmax and ymax < bymax:
        return True
    else:
        kxmin = max(bxmin, xmin)
        kymin = max(bymin, ymin)
        kxmax = min(bxmax, xmax)
        kymax = min(bymax, ymax)
        iou = (xmax - xmin) * (ymax - ymin) / ((kxmax - kxmin) * (kymax - kymin))
        if iou >= min_iou:
            return True
        else:
            return False


def search_box(bound, boxes, is_inside_fn=is_inside, min_iou=0.7):
    return [box for box in boxes if is_inside_fn(min_iou, bound, box)]


class Grid(object):
    def __init__(self, cover_shape, split_shape):
        image_h, image_w = cover_shape
        s_h, s_w = split_shape
        self._cover_shape = cover_shape
        self._split_shape = split_shape
        self._grid_w = image_h / s_h
        self._grid_h = image_w / s_w
        assert isinstance(image_h % s_h, int) and isinstance(image_w % s_w, int), \
            "The shape of image must be an integral multiple of grid."

    def get_bound_by_id(self, i, j):
        """get bound's  coordinates by index

        :param i: Index of row
        :param j: Index of column
        :return: A tuple (xmin, ymin, xmax, ymax)
        """
        mi, mj = self._split_shape
        assert i < mi and j < mj, "Index is out of range."

        xmin, ymin = (i * self._grid_w, j * self._grid_h)
        xmax = xmin + self._grid_w
        ymax = ymin + self._grid_h

        return xmin, ymin, xmax, ymax


class Spliter(object):
    def __init__(self, split_shape, example):
        self._grid = None
        self._example = example
        self._channel = None
        self._split_shape = split_shape

        img = self._example.image
        h = 1
        w = 1

        if len(img.shape) == 2:
            h, w = img.shape
            self._channel = 1
        if len(img.shape) == 3:
            h, w, self._channel = img.shape

        self._grid = Grid((h, w), self._split_shape)

    def get_subimage_by_id(self, i, j):
        xmin, ymin, xmax, ymax = self._grid.get_bound_by_id(i, j)
        if self._channel == 1:
            return self._example.image[ymin:ymax, xmin:xmax]
        elif self._channel == 3:
            return self._example.image[ymin:ymax, xmin:xmax, :]

    def get_boxes_by_id(self, i, j):
        bound = self._grid.get_bound_by_id(i, j)
        boxes = search_box(bound, self._example.bboxes, min_iou=0.7)
        return boxes

    def get_example_by_id(self, i, j):
        example = Example()
        example.image = self.get_subimage_by_id(i, j)
        example.bboxes = self.get_boxes_by_id(i, j)
        return example
