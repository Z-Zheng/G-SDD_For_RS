# G-SSD model tutorial
import numpy as np
from skimage.io import imread
from app.core.loader import load_config
from app.core.main_model import Main_Model
from app.core.grid import fake_process

# load protobuf file defining basic sdd_inception_v2 model
model_config, _, __ = load_config()
# Construct our G-SSD model
model = Main_Model(model_config)
# define graph
model.define_graph()

# start a session for the graph and load some weights which includes
# all parameters of basic ssd_inception_v2 and partial paramters of
# inception_v2.The latter just provides final layer's paramters including
# convolution kernel and biase.
model.open_session()
model.load_ckpt_to_graph(ssd_ckpt='ssd300_model/model.ckpt-67344',
                         inception_ckpt='inception_model/model.ckpt-0')

# read an imgae [m,n,3]
img = imread('image_test/013.jpg')
# expand dimension of this image to [1,m,n,3]
img = np.expand_dims(img, axis=0)
# grid classify
img = fake_process(img)
prob, fmp = model.grid_classify(img)
prob = prob[0]
if model.is_detect(prob):
    blob = model.detect(fmp)
    print(blob)
