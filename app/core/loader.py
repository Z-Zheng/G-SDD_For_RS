"""load resources to memory
function [load_model_to_graph]:load model architecture and paramters to build graph.
function [load_label_map_to_build_category_idx]:load label_map.pbtxt and convert it
 to category index for visualization.
"""
import tensorflow as tf
from app import config
from object_detection.utils import label_map_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


def load_config():
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(config.FLAGS.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    model_config = pipeline_config.model
    train_config = pipeline_config.train_config
    input_config = pipeline_config.train_input_reader

    return model_config, train_config, input_config


def load_ckpt_to_session(session, var_list, ckpt_path):
    saver = tf.train.Saver(var_list)
    saver.restore(session, ckpt_path)


def load_label_map_to_build_category_idx():
    label_map = label_map_util.load_labelmap(config.FLAGS.path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=config.FLAGS.num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return category_index
