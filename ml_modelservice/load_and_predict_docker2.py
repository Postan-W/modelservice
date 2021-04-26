import json
from pyspark.sql import SparkSession
import argparse
from define_spark import spark
sc = spark.sparkContext
from pyspark.sql import functions as f

"""
参数一：unlabeled_data，测试数据集，类型为data:pyspark.sql.dataframe，必须包含训练model_path对应的模型的训练集中的特征列的所有字段
参数二：model_path, 已拟合模型的路径，类型为字符串，对应模型必须为PipelineModel

返回值：
        result,字典，键包含：prediction
        result["prediction"]，模型预测结果，类型为pyspark.sql.dataframe.DataFrame,其中prediction列为预测结果

使用示例：      
>>>data1 = spark.createDataFrame([[0.04,10.767,0],[0.03,10.567,0],[0.02,10.7,1]],['CV','STD','IS_OUTNET'])
>>>data1.show()
+----+------+---------+
|  CV|   STD|IS_OUTNET|
+----+------+---------+
|0.04|10.767|        0|
|0.03|10.567|        0|
|0.02|  10.7|        1|
+----+------+---------+
>>>model_path = r'hdfs://172.16.41.26:9000/649ca1d097924be3a5b49abedb249c0b/dataSet/devdatasetdef1083/1557910468/model'
>>>import sys
>>>sys.path.append(r'F:\5\12306-master\12306-master\a_dataiiku\my_design')
>>>from load_and_predict import Predict
>>>result = Predict.predict_by_model(data1, model_path)
>>>result['prediction'].show()
+----+------+---------+----------+
|  CV|   STD|IS_OUTNET|prediction|
+----+------+---------+----------+
|0.04|10.767|        0|       0.0|
|0.03|10.567|        0|       0.0|
|0.02|  10.7|        1|       0.0|
+----+------+---------+----------+
>>>prediction = result['prediction'].select('prediction')
>>>prediction.show()
+----------+
|prediction|
+----------+
|       0.0|
|       0.0|
|       0.0|
+----------+
"""


def prob_to_label(df, prediction_col='prediction',predict_label_col='predict_label'):
    """
    将df中的概率列，转成原值标签，生成预测标签列
    :param df: pyspark.sql.DataFrame，H20模型transform后的DataFrame
    :param prediction_col: str,原预测的概率列
    :param predict_label_col: str,预测出的标签值列名
    :return: pyspark.sql.DataFrame 生成预测标签列的数据集
    """
    schema = df.schema

    p_schema = schema[-1]
    fields = p_schema.json()
    json_ = p_schema.json()
    dict1 = json.loads(json_)
    list_ = dict1['type']['fields']
    list1 = [(i, list_[i]['name']) for i in range(len(list_))]
    prob_dict = dict(list1)
    # prob_json = json.dumps(prob_dict)
    prob_json = str(prob_dict)

    def func(array, json_):
        # dict_ = json.loads(json_)
        dict_ = eval(json_)
        max_p = max(array)
        argmax = array.index(max_p)
        p_name = dict_[argmax]
        return float(p_name[1:])

    udf_ = f.udf(func)

    df = df.withColumn(predict_label_col, udf_(f.col(prediction_col), f.lit(prob_json)))
    return df


class Predict:

    def __init__(self):
        super().__init__()
        self.json_data = None
        self.json_type = None
        self.model_path = None
        self.parser = argparse.ArgumentParser()

    def setParams(self):
        self.parser.add_argument("--json_data", help="display a model path,refer to a fitted model", type=str)
        # self.parser.add_argument("--json_type", help="display a output path,refer to a fitted model", type=str)
        self.parser.add_argument("--model_path", help="display a output path,refer to a fitted model", type=str)
        self.args = self.parser.parse_args()
        # print('000------------------------')

        # self.after_add_argument()

        self.json_data = self.args.json_data
        # self.json_type = self.args.json_type
        self.model_path = self.args.model_path
        # print(self.json_type)

    def predict(self):
        # unlabeled_data = self.read_data_by_path()
        return self.predict_by_model(self.json_data, self.model_path)

    @staticmethod
    def predict_by_model(json_data,model):
        """
        加载模型，并对数据集作预测
        :param unlabeled_data: pyspark.sql.DataFrame 至少包含训练集的所有特征列
        :param model_path: str, 模型路径，用于加载获得PipelineModel,仅针对hdfs中模型路径为model的情形
        :return: dict,包含键prediction，值为pyspark.sql.DataFrame类型，字段prediction为预测结果
        """
        data = spark.createDataFrame((json_data))
        columns = data.columns
        # 若传入的unlabeled_data包含prediction列，则删除
        if 'prediction' in columns:
            columns = columns[:-1]
            print(columns)
            data = data.select(*columns)
        print('unlabeled data display---------------------------------------------')
        data.show()
        try:
            prediction = model.transform(data)  # 进行预测
            prediction = prediction.drop("features", "indexedFeatures", "rawPrediction", "indexedLabel",
                                         "detailed_prediction")  # 去掉一些向量列，否则无法保存
            # 如果包含解码器，那么就把"prediction"去掉，然后将"prediction_str"重命名为"prediction"
            if model.stages.__len__() == 3:
                prediction = prediction.drop("prediction")
                prediction = prediction.withColumnRenamed("prediction_str", "prediction")
        except:
            prediction = model.transform(data)  # 进行预测
            prediction = prediction.drop("detailed_prediction")  # 去掉一些向量列，否则无法保存
            prediction = prob_to_label(prediction, prediction_col='prediction', predict_label_col='prediction_label')
        return eval(prediction.toPandas().to_json(orient='records'))


def main(algorithm_name):
    pred_alg = eval("{}()".format(algorithm_name))
    pred_alg.setParams()
    return pred_alg.predict()

# if __name__ == '__main__':
#     main('Predict')


# {"displayName":"myds","fieldDefs":[{"code":"ZERO","name":"ZERO","displayName":"ZERO","ord":0,"originalType":12,"originalTypeName":"VARCHAR","typeClassName":"java.math.BigInteger","fieldType":"DIMENSION","exprHasAggr":false,"dimEnabled":false,"cascade":false},{"code":"MEAN","name":"MEAN","displayName":"MEAN","ord":1,"originalType":12,"originalTypeName":"VARCHAR","typeClassName":"java.math.BigDecimal","fieldType":"DIMENSION","exprHasAggr":false,"dimEnabled":false,"cascade":False},{"code":"STD","name":"STD","displayName":"STD","ord":2,"originalType":12,"originalTypeName":"VARCHAR","typeClassName":"java.math.BigDecimal","fieldType":"DIMENSION","exprHasAggr":false,"dimEnabled":false,"cascade":false},{"code":"CV","name":"CV","displayName":"CV","ord":3,"originalType":12,"originalTypeName":"VARCHAR","typeClassName":"java.math.BigDecimal","fieldType":"DIMENSION","exprHasAggr":false,"dimEnabled":false,"cascade":false},{"code":"INC","name":"INC","displayName":"INC","ord":4,"originalType":12,"originalTypeName":"VARCHAR","typeClassName":"java.math.BigDecimal","fieldType":"DIMENSION","exprHasAggr":false,"dimEnabled":false,"cascade":false},{"code":"OPP","name":"OPP","displayName":"OPP","ord":5,"originalType":12,"originalTypeName":"VARCHAR","typeClassName":"java.math.BigInteger","fieldType":"DIMENSION","exprHasAggr":false,"dimEnabled":false,"cascade":false},{"code":"CS","name":"CS","displayName":"CS","ord":6,"originalType":12,"originalTypeName":"VARCHAR","typeClassName":"java.math.BigInteger","fieldType":"DIMENSION","exprHasAggr":false,"dimEnabled":false,"cascade":false},{"code":"IS_OUTNET","name":"IS_OUTNET","displayName":"IS_OUTNET","ord":7,"originalType":12,"originalTypeName":"VARCHAR","typeClassName":"java.math.BigInteger","fieldType":"DIMENSION","exprHasAggr":false,"dimEnabled":false,"cascade":false},{"code":"prediction","name":"prediction","displayName":"prediction","ord":8,"originalType":12,"originalTypeName":"VARCHAR","typeClassName":"java.math.BigInteger","fieldType":"DIMENSION","exprHasAggr":false,"dimEnabled":false,"cascade":false}]}



"""
--inputpath
hdfs://172.16.41.26:9000/649ca1d097924be3a5b49abedb249c0b/dataSet/devdatasetdef1083/1558945988/test_data
--output
hdfs://172.16.41.26:9000/649ca1d097924be3a5b49abedb249c0b/dataSet/devdatasetdef1083
--model_path
hdfs://172.16.41.26:9000/649ca1d097924be3a5b49abedb249c0b/dataSet/devdatasetdef1083/1557910468/model
"""

"""
--json_data
[{'ZERO':29,'MEAN':0.067,'STD':0.365,'CV':5.477,'INC':0.0,'OPP':2,'CS':0,'IS_OUTNET':1},{'ZERO':1,'MEAN':5.8,'STD':3.188,'CV':0.55,'INC':1.313,'OPP':100,'CS':0,'IS_OUTNET':1}]
--model_path
/Users/mengdihao/python-data/dtc/model
"""