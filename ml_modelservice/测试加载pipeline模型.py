from pyspark.ml import PipelineModel
model_path = ""
try:
    model = PipelineModel.load(model_path)
    print("加载成功")
except Exception as e:
    print(e)

