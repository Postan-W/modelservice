import os
from pyspark.ml import PipelineModel
model_path = "./decision_tree_model/model"

model = PipelineModel(model_path)