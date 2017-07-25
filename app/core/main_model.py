import tensorflow as tf
from object_detection.builders import model_builder
from app.core import feature_extractor
from app.core import detector
from app.core import grid_classifier
from app.core import loader


def _image_tensor_input_placeholder():
    return tf.placeholder(dtype=tf.uint8,
                          shape=(1, None, None, 3),
                          name='image_tensor')


class Main_Model(object):
    def __init__(self, model_config):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._init_op = tf.global_variables_initializer()
            self._model = model_builder.build(model_config, is_training=False)
        self._feature_extractor = feature_extractor.Feature_Extractor(self)
        self._grid_classifier = grid_classifier.Grid_Classifier(self)
        self._detector = detector.Detector(self)
        self._sess = None
        self._input_op = None
        self._output_op = None
        self._feature_maps_dict_op = None
        self._is_open_sess = False
        self._is_def_graph = False

    def __del__(self):
        if self._is_open_sess:
            self._sess.close()
        self._sess = None
        self._graph = None

    # those followed three methods will be replaced by 'with'
    def open_session(self):
        if not self._is_open_sess:
            assert self._is_def_graph, "The graph is not defined.Please define it first."
            self._sess = tf.Session(graph=self._graph)
            self._sess.run(self._init_op)
            self._is_open_sess = not self._is_open_sess

    def reset_session(self):
        if self._is_open_sess:
            self._sess.close()
            self._sess = None
            self._is_open_sess = not self._is_open_sess
        self.open_session()

    def close_session(self):
        if self._is_open_sess:
            self._sess.close()
            self._sess = None
            self._is_open_sess = not self._is_open_sess

    @property
    def internal_model(self):
        return self._model

    @property
    def graph(self):
        return self._graph

    @property
    def session(self):
        return self._sess

    def _build_input(self, input_op):
        self._input_op = input_op

    def _build_output(self, output_op):
        self._output_op = output_op

    def define_graph(self):
        if self._is_def_graph:
            return
        self._is_def_graph = True
        with self._graph.as_default():
            image_tensor = _image_tensor_input_placeholder()
            self._build_input(image_tensor)

            preprocessed_image = self._model.preprocess(tf.to_float(image_tensor))
        # define operation of extracting feature
        self._feature_extractor.define(preprocessed_image)
        mixed_5c_feature, feature_maps = self._feature_extractor.output
        # define operation of grid classify
        self._grid_classifier.define(mixed_5c_feature)
        probability = self._grid_classifier.output
        # define operation of detecting
        self._detector.define(feature_maps)
        boxes, scores, classes, num_detections = self._detector.output

        self._build_output([boxes, scores, classes, num_detections])

        self._feature_maps_dict_op = {}
        for feature_map in feature_maps:
            self._feature_maps_dict_op['%s:0' % feature_map.op.name] = feature_map

    def grid_classify(self, image, need_classify=True):
        stage_1_feed_dict = {
            self._input_op: image
        }
        if need_classify:
            probability, feature_map_dict = self.session.run([self._grid_classifier.output, self._feature_maps_dict_op],
                                                             feed_dict=stage_1_feed_dict)
            return probability, feature_map_dict
        else:
            feature_map_dict = self.session.run(self._feature_maps_dict_op,
                                                feed_dict=stage_1_feed_dict)
            return feature_map_dict

    def detect(self, feature_map_dict):
        stage_2_feed_dict = feature_map_dict
        (boxes, scores, classes, num_detections) = self.session.run(
            self._detector.output, feed_dict=stage_2_feed_dict)
        return boxes, scores, classes, num_detections

    def is_detect(self, probability):
        return probability[0] < probability[1]

    def get_grid_classifier(self):
        return self._grid_classifier

    def get_feature_extractor(self):
        return self._feature_extractor

    def get_detector(self):
        return self._detector

    def write_graph_to_file(self, logdir):
        fw = tf.summary.FileWriter(logdir)
        fw.add_graph(self._graph)
        fw.close()

    def load_ckpt_to_graph(self, inception_ckpt=None, ssd_ckpt=None):
        assert self._is_def_graph, "The graph is not defined.Please define it first."

        with self.graph.as_default():
            grid_classifier_var = []
            ssd_feature_extractor_var = []
            ssd_box_predictor_var = []
            for var in tf.global_variables():
                if var.op.name.startswith("Inception"):
                    grid_classifier_var.append(var)
                elif var.op.name.startswith(self.internal_model._extract_features_scope):
                    ssd_feature_extractor_var.append(var)
                else:
                    ssd_box_predictor_var.append(var)
            ssd_var = ssd_feature_extractor_var + ssd_box_predictor_var

            if isinstance(ssd_ckpt, str):
                loader.load_ckpt_to_session(self._sess, ssd_var, ssd_ckpt)
            if isinstance(inception_ckpt, str):
                loader.load_ckpt_to_session(self._sess, grid_classifier_var, inception_ckpt)
