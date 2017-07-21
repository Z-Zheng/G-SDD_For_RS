from skimage.io import imread, imsave
from dataset.label_map import label_map


class BBox(object):
    def __init__(self, coord, category, blurred=0):
        self.coord = coord
        self.category = category
        self.blurred = blurred


def write_image_to_file(image, filename, img_format='.png'):
    imsave("{0}.{1}".format(filename, img_format), image)


def write_boxes_to_file(boxes, save_path):
    format = '{0} {1} {2} {3} {4} {5} {6} {7} {8}\n'
    with open(save_path, 'w') as f:
        for box in boxes:
            xmin, ymin, xmax, ymax = box.coord
            category_str = label_map.label_map_str[box.category]
            record = format.format(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, category_str)
            f.write(record)


class Example(object):
    def __init__(self):
        self.bboxes = []
        self.image = None

    def write_to_file(self, image_filename, image_format, boxes_save_path):
        write_image_to_file(self.image, image_filename, image_format)
        write_boxes_to_file(self.bboxes, boxes_save_path)


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
