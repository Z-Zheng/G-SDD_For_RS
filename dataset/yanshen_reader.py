from skimage.io import imread
from dataset.label_map import label_map


class BBox(object):
    def __init__(self, coord, category, blurred=0):
        self.coord = coord
        self.category = category
        self.blurred = blurred


class Example(object):
    def __init__(self):
        self.bboxes = []
        self.image = None


def parse_line(line):
    v = line.split(' ')
    # position
    xmin = v[0]
    ymin = v[1]
    xmax = v[4]
    ymax = v[5]
    # class  type: int
    category = label_map[v[8]]

    blurred = int(len(v) == 10)

    bbox = BBox([xmin, ymin, xmax, ymax], category, blurred)

    return bbox


def parse_label(label_filename):
    with open(label_filename, 'r') as f:
        lines = f.read().split('\n')[:-1]
    bboxes = map(parse_line, lines)

    return list(bboxes)


def read(image_filename, label_filename, read_img=True):
    img = None
    if read_img:
        img = imread(image_filename)
    bboxes = parse_label(label_filename)

    example = Example()
    example.image = img
    example.bboxes = bboxes

    return example
