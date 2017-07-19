from app.core.loader import load_config
from app.core.main_model import Main_Model

model_config, _, __ = load_config()

model = Main_Model(model_config)
model.define_graph()

model.write_graph_to_file('model_w')

