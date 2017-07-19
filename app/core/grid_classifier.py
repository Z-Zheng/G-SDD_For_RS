import tensorflow as tf
from slim.nets import inception_utils

slim = tf.contrib.slim


def inception_v2_classify(inputs,
                          num_classes=2,
                          is_training=True,
                          dropout_keep_prob=0.8,
                          depth_multiplier=1.0,
                          prediction_fn=slim.softmax,
                          spatial_squeeze=True,
                          reuse=None,
                          scope='InceptionV2'):
    """Inception v2 model for classification.

    Constructs an Inception v2 network for classification as described in
    http://arxiv.org/abs/1502.03167.

    The default image size used to train this network is 224x224.

    Args:
      inputs: a tensor of shape [batch_size, ksize, ksize, channels].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: the percentage of activation values that are retained.
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when depth_multiplier < 1, and not an active constraint when
        depth_multiplier >= 1.
      depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      prediction_fn: a function to get predictions out of logits.
      spatial_squeeze: if True, logits is of shape [B, C], if false logits is
          of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, num_classes]
      end_points: a dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: if final_endpoint is not set to one of the predefined values,
                  or depth_multiplier <= 0
    """
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    # Final pooling and prediction
    with tf.variable_scope(scope, 'InceptionV2', [inputs, num_classes],
                           reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            end_points = {}

            with tf.variable_scope('Logits'):
                kernel_size = _reduced_kernel_size_for_small_input(inputs, [16, 16])
                net = slim.avg_pool2d(inputs, kernel_size, padding='VALID',
                                      scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                # 1 x 1 x 1024
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.

    If the shape of the input images is unknown at graph construction time this
    function assumes that the input images are is large enough.

    Args:
      input_tensor: input tensor of size [batch_size, height, width, channels].
      kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

    Returns:
      a tensor with the kernel size.

    TODO(jrru): Make this function work with unknown shapes. Theoretically, this
    can be done with the code below. Problems are two-fold: (1) If the shape was
    known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
    handle tensors that define the kernel size.
        shape = tf.shape(input_tensor)
        return = tf.pack([tf.minimum(shape[1], kernel_size[0]),
                          tf.minimum(shape[2], kernel_size[1])])

    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]
    return kernel_size_out


class Grid_Classifier(object):
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

            _, probability_op = inception_v2_classify(self._input)
            probability_op = probability_op['Predictions']
            self._output = probability_op
