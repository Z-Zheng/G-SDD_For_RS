import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
from app.core.loader import load_config, load_label_map_to_build_category_idx
from app.core.main_model import Main_Model
from app import config
from app.core.grid import Spliter
from dataset.yanshen_reader import Example
from app.submit import Submit
from object_detection.utils import visualization_utils as vis_util
import matplotlib.pyplot as plt

STD_SHAPE = [[1000, 1000], [4000, 4000]]
SPLIT_SHAPE = [(4, 4), (8, 8)]

category_index = load_label_map_to_build_category_idx()


def vis(image, boxes, scores, classes, category_index):
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    IMAGE_SIZE = (12, 8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image)


def select_shape(h, w):
    sum_data = h * w
    small_shape = STD_SHAPE[0]
    large_shape = STD_SHAPE[1]
    if abs(sum_data - small_shape[0] * small_shape[1]) > abs(sum_data - large_shape[0] * large_shape[1]):
        return large_shape, 1
    else:
        return small_shape, 0


def preprocess(image):
    h = 0
    w = 0
    c = 0
    is_single = True
    if len(image.shape) == 2:
        h, w = image.shape
    if len(image.shape) == 3:
        h, w, c = image.shape
        is_single = not is_single

    assert h != 0 and w != 0, ''

    new_shape, size_idx = select_shape(h, w)
    nh, nw = new_shape
    if is_single:
        return resize(image, (nh, nw)), nh / h, nw / w, size_idx
    else:
        return resize(image, (nh, nw, c)), nh / h, nw / w, size_idx


def postprocess_rel2abs(bound, boxes, mh, mw):
    bxmin, bymin, bxmax, bymax = bound
    nboxes = np.copy(boxes)
    # shift
    nboxes[:, 0] += bxmin
    nboxes[:, 1] += bymin
    nboxes[:, 2] += bxmax
    nboxes[:, 3] += bymax
    # scale
    nboxes[:, 0] *= mw
    nboxes[:, 2] *= mw

    nboxes[:, 1] *= mh
    nboxes[:, 3] *= mh

    return nboxes


def postprocess(boxes, scores, classes, num_detections, min_conf=0.5):
    blobs = [(boxes[i], scores[i], classes[i])
             for i in range(num_detections) if scores[i] >= min_conf]
    nboxes = []
    nscores = []
    nclasses = []
    for blob in blobs:
        box, score, category = blob
        nboxes.append(box)
        nscores.append(nscores)
        nclasses.append(nclasses)
    return np.asarray(nboxes), np.asarray(nscores), np.asarray(nclasses), len(blobs)


def read_all():
    for filename in os.listdir(config.FLAGS.image_dir):
        abs_fn = os.path.join(config.FLAGS.image_dir, filename)
        yield imread(abs_fn), filename


def main():
    inputs = read_all()
    model_config, _, __ = load_config()
    model = Main_Model(model_config)
    model.define_graph()
    model.open_session()
    model.load_ckpt_to_graph(ssd_ckpt=config.FLAGS.path_to_ckpt,
                             inception_ckpt=None)
    submit = Submit()
    submit.open_files()
    for image, filename in read_all():
        preprocessed_image, mh, mw, size_idx = preprocess(image)
        example = Example()
        example.image = preprocessed_image
        (mi, mj) = SPLIT_SHAPE[size_idx]
        spliter = Spliter((mi, mj), example)
        for i in range(mi):
            for j in range(mj):
                subimage = spliter.get_subimage_by_id(i, j)
                subimage = np.expand_dims(subimage, axis=0)
                subimage = np.expand_dims(subimage, axis=3)
                # [1,500,500,3]
                subimage = np.concatenate((subimage, subimage, subimage), axis=3)
                fmp = model.grid_classify(subimage, need_classify=False)
                (boxes, scores, classes, num_detections) = model.detect(fmp)
                (boxes, scores, classes, num_detections) = postprocess(boxes, scores, classes, num_detections)
                # Tensor [100,4]
                boxes = np.squeeze(boxes)
                # Tensor [100]
                scores = np.squeeze(scores)
                # Tensor [100]
                classes = np.squeeze(classes)
                bound = spliter.get_bound_by_id(i, j)
                # coordnate transform
                boxes = postprocess_rel2abs(bound, boxes, mh, mw)
                # per object
                for k in range(num_detections):
                    xmin, ymin, xmax, ymax = boxes[k]
                    score = scores[k]
                    category = classes[k]
                    submit.append(filename, score, (xmin, ymin, xmax, ymax), category)
    submit.write_to_file()
    submit.close_files()


if __name__ == '__main__':
    main()
