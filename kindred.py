# https://docs.allennlp.org/models/main/
from allennlp_models import pretrained

model = pretrained.load_predictor("coref-spanbert")

test = model.predict_instance("This is a test. Do you think it will work?")

print(test)
