import tensorflow as tf

scope_names = ['Conv2d_0a_1x1', 'Conv2d_0b_3x3', 'Conv2d_0c_3x3', 'Conv2d_0b_1x1']


class Feature_Extractor(object):
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

            with tf.variable_scope(None, self._main_model.internal_model._extract_features_scope,
                                   [self.input]):
                feature_maps = self._main_model.internal_model._feature_extractor.extract_features(
                    self.input)

            mixed_5c_branchs = []
            for idx, scope_name in enumerate(scope_names):
                tensor_name = '%s/Branch_%d/%s/Relu6:0' % (
                    'FeatureExtractor/InceptionV2/InceptionV2/Mixed_5c', idx, scope_name)
                mixed_5c_branchs.append(self._main_model.graph.get_tensor_by_name(tensor_name))
            # for 224x224 image-> 7x7x1024
            mixed_5c_feature = tf.concat(mixed_5c_branchs, axis=3)
            self._output = (mixed_5c_feature, feature_maps)
