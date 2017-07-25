"""Detector for a batch of images fixed size.
"""
import tensorflow as tf
import numpy as np


class Detector(object):
    def __init__(self, main_model):
        self._main_model = main_model
        self._input = None
        self._output = None

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    def define(self, input_op):
        with self._main_model.graph.as_default():
            self._input = input_op
            feature_map_spatial_dims = self._main_model.internal_model._get_feature_map_spatial_dims(self._input)
            self._main_model.internal_model._anchors = self._main_model.internal_model._anchor_generator.generate(
                feature_map_spatial_dims)
            (box_encodings, class_predictions_with_background
             ) = self._main_model.internal_model._add_box_predictions_to_feature_maps(self._input)
            predictions_dict = {
                'box_encodings': box_encodings,
                'class_predictions_with_background': class_predictions_with_background,
                'feature_maps': self._input
            }
            label_id_offset = 1
            detections = self._main_model.internal_model.postprocess(predictions_dict)
            boxes = detections.get('detection_boxes')
            scores = detections.get('detection_scores')
            classes = detections.get('detection_classes') + label_id_offset
            num_detections = detections.get('num_detections')
            self._output = (boxes, scores, classes, num_detections)
