from app.core import grid
from dataset.yanshen_reader import read, Example, BBox
import random
import string
import os
from skimage.transform import resize


def preprocess(example, std_shape=(4000, 4000)):
    h, w = (0, 0)
    if len(example.image.shape) == 3:
        h, w, c = example.image.shape
    if len(example.image.shape) == 2:
        h, w = example.image.shape
    mh = std_shape[0] / h
    mw = std_shape[1] / w

    nimage = resize(example.image, std_shape)
    nexample = Example()
    nexample.image = nimage
    for box in example.bboxes:
        bxmin, bymin, bxmax, bymax = box.coord
        coord = (bxmin * mw, bymin * mh, bxmax * mw, bymax * mh)
        nbox = BBox(coord, box.category, box.blurred)
        nexample.bboxes.append(nbox)

    return nexample


def grid_cut(image_filename, label_filename, save_dir, split_shape=(16, 16)):
    example = read(image_filename, label_filename)
    example = preprocess(example, std_shape=(4000, 4000))
    spliter = grid.Spliter(split_shape, example)
    mi, mj = split_shape
    # define directory structure
    image_dir_name = 'image'
    label_dir_name = 'label'
    if not os.path.exists(os.path.join(save_dir, image_dir_name)):
        os.makedirs(os.path.join(save_dir, image_dir_name))
    if not os.path.exists(os.path.join(save_dir, label_dir_name)):
        os.makedirs(os.path.join(save_dir, label_dir_name))

    for i in range(mi):
        for j in range(mj):
            cur_example = spliter.get_example_by_id(i, j)
            # get random prefix
            prefix = ''.join(random.choice(string.ascii_letters) for x in range(8))
            image_filename = "{0}_grid_{1}{2}".format(prefix, i, j)
            label_filename = "{0}_grid_{1}{2}.txt".format(prefix, i, j)
            cur_example.write_to_file(image_filename=os.path.join(save_dir, image_dir_name, image_filename),
                                      image_format='.png',
                                      boxes_save_path=os.path.join(save_dir, label_dir_name, label_filename))
            print("[{2}][{3}]:{0} has been saved in {1}\n".format(image_filename, save_dir, i, j))
            print("[{2}][{3}]:{0} has been saved in {1}\n".format(label_filename, save_dir, i, j))


if __name__ == '__main__':
    grid_cut(r'unit_test/1.tiff', r'unit_test/1.txt', r'unit_test/save')
