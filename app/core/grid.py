"""split remote sensing image into many patch fixed shape by grid

"""
import numpy as np
from skimage import io as imageio
from dataset.yanshen_reader import Example, BBox


def fake_process(image):
    return image

def overlap( x1,  w1,  x2,  w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    if l1 > l2:
        left = l1
    else:
        left = l2
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    if r1 < r2:
        right = r1
    else:
        right = r2
    return right - left

def box_intersection(x1,y1,w1,h1,x2,y2,w2,h2 ):

    w = overlap(x1, w1, x2, w2)
    h = overlap(y1, h1, y2, h2)
    if w < 0 or h < 0:
        return 0
    area = w*h
    return area

def overlap_cord( x1,  w1,  x2,  w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    if l1 > l2:
        left = l1
    else:
        left = l2
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    if r1 < r2:
        right = r1
    else:
        right = r2
    return int(right),int(left)

def box_intersection_cord(x1,y1,w1,h1,x2,y2,w2,h2 ):
    right, left = overlap_cord(x1, w1, x2, w2)
    bottom, top = overlap_cord(y1, h1, y2, h2)
    return left,top,right,bottom

def is_inside(min_iou, bound, box):
    bxmin, bymin, bxmax, bymax = bound
    bcx = (bxmax + bxmin ) / 2
    bcy = (bymax + bymin) / 2
    xmin, ymin, xmax, ymax = box.coord
    bbx = (xmax + xmin ) / 2
    bby = (ymax + ymin) / 2
    if bxmin < xmin and bymin < ymin and xmax < bxmax and ymax < bymax:
        return True
    else:
        inter = box_intersection(bcx, bcy, bxmax - bxmin, bymax - bymin, bbx, bby, xmax - xmin, ymax - ymin)
        if xmax - xmin == 0 or ymax - ymin == 0:
            print(xmax,',',xmin,',',ymax,',',ymin)
        barea = ( xmax - xmin)* (ymax - ymin)
        iou = inter / barea
        #print(iou)
        if iou >= min_iou:
            left, top, right, bottom = box_intersection_cord(bcx, bcy, bxmax - bxmin, bymax - bymin, bbx, bby, xmax - xmin, ymax - ymin)
            if left == bxmin:
                left = left + 1
            elif top == bymin:
                top = top + 1
            elif right == bxmax:
                right = right - 1
            elif bottom == bymax:
                bottom = bottom - 1
            box.coord = [left, top, right, bottom]
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

        xmin, ymin = (j * self._grid_w, i * self._grid_h)
        xmax = xmin + self._grid_w
        ymax = ymin + self._grid_h

        return xmin, ymin, xmax, ymax


def postprocess(bound, boxes):
    bxmin, bymin, bxmax, bymax = bound
    nboxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box.coord
        nbox = BBox((xmin - bxmin, ymin - bymin, xmax - bxmin, ymax - bymin),
                    box.category, box.blurred)
        nboxes.append(nbox)
    return nboxes


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
        x_min, y_min, x_max, y_max = self._grid.get_bound_by_id(i, j)
        xmin = int(x_min)
        ymin = int(y_min)
        xmax = int(x_max)
        ymax = int(y_max)
        if self._channel == 1:
            return self._example.image[ymin:ymax, xmin:xmax]
        elif self._channel == 3:
            return self._example.image[ymin:ymax, xmin:xmax, :]

    def get_boxes_by_id(self, i, j):
        bound = self._grid.get_bound_by_id(i, j)
        boxes = search_box(bound, self._example.bboxes, min_iou=0.7)
        return postprocess(bound, boxes)

    def get_example_by_id(self, i, j):
        example = Example()
        example.image = self.get_subimage_by_id(i, j)
        example.bboxes = self.get_boxes_by_id(i, j)
        return example
