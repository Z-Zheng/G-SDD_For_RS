import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('path_to_ckpt', '',
                    'Path to a checkpoint file')
flags.DEFINE_string('path_to_labels', '',
                    'Path to a label file')
flags.DEFINE_integer('num_classes', 10,
                     'number of category')

flags.DEFINE_string('pipeline_config_path', 'ssd_inception_v2_pets.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')

FLAGS = flags.FLAGS
