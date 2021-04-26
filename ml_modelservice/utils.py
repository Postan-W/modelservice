import datetime
import json
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import *

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("mapreduce.output.fileoutputformat.compress", "false") \
    .config("spark.broadcast.compress", "false") \
    .config("spark.sql.parquet.compression.codec", "uncompressed") \
    .enableHiveSupport() \
    .getOrCreate()

sc = spark.sparkContext


def timmer(func):
    """
    装饰器，计算时间
    :param func:
    :return:
    """

    def wapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        res = func(*args, **kwargs)
        stop_time = datetime.datetime.now()
        print("{func}函数的运行时间为{time}".format(func=func.__name__, time=stop_time - start_time))
        return res

    return wapper


def get_label_class_num(data, label_col):
    """
    计算数据集label列类数，针对分类模型
    :param data: 待训练数据集
    :param label_col: label列
    :return: 返回label列类数
    """
    data_label = data.select(label_col)
    # data_label = data_label.withColumn(label_col, data_label[label_col])
    label_num = data_label.dropDuplicates().count()
    # distinct_label = data_label.rdd.distinct().collect()
    # label_num = len(distinct_label)
    # distinct_label = [distinct_label[i][0] for i in range(label_num)]

    if label_num == 2:
        class_type = "binary"
    elif label_num > 2:
        class_type = "multi-class"
    else:
        class_type = "one-class"
    return class_type


# def get_label_class_num(data, label_col):
#     """
#     计算数据集label列类数，针对分类模型
#     :param data: 数据集
#     :param label_col: label列名字
#     :return: 返回label列类数
#     """
#     data_label = data.select(label_col)
#     label_num = data_label.dropDuplicates().count()
#     if label_num == 2:
#         class_type = "binary"
#     elif label_num > 2:
#         class_type = "multi-class"
#     else:
#         class_type = "one-class"
#     return class_type


def check_null(df):
    """
    看一个dataframe中是否包含null，只要有一个null，就返回Ture
    :param df: dataframe
    :return: boolen True or False
    """
    assert isinstance(df, DataFrame)
    df_dropna = df.dropna(how="any")  # 删除null时采用只要有null就删除该行数据，
    return df.count() != df_dropna.count()


def prob_to_label1(df, prediction_col="prediction"):
    """
    h2o中概率转成类别
    :param df:
    :param prediction_col:概率列名字
    :return:
    """
    schema = df.select(prediction_col).schema[0]
    json_data = json.loads(schema.json())
    number = len(json_data["type"]["fields"])  # 如果是多分类，列表嵌套列表，number为1，需要先做转换
    if number < 2:
        pull_udf = udf(lambda x: x[0])  # [[0.1,0.3,0.6]] -> [0.1,0.3,0.6]
        df = df.withColumn(prediction_col, pull_udf(df[prediction_col]))
    index_udf = udf(lambda x: float(x.index(max(x))), returnType=FloatType())  # [0.1,0.3,0.6]->2
    df = df.withColumn(prediction_col, index_udf(df[prediction_col]))
    return df


def get_columns(label_col=None, feature_cols=None, weight_col=None):
    """
    将label_col和feature_cols做成一个dataframe
    +------------+------------+------------+---------+
    |        ZERO|         STD|          CV|IS_OUTNET|
    +------------+------------+------------+---------+
    |feature_cols|feature_cols|feature_cols|label_col|
    +------------+------------+------------+---------+
    :param label_col:
    :param feature_cols:
    :return: df
    """
    names = ["feature_cols"] * len(feature_cols)
    cols = [i for i in feature_cols]

    if label_col is not None:
        names.append("label_col")
        cols.append(label_col)
    if weight_col != '' and weight_col is not None and weight_col != 'None':
        names.append("weight_col")
        cols.append(weight_col)
    df = spark.createDataFrame([names], schema=cols)
    return df


def train_test_split(data, test_data_pro, seed=1234):
    """
    测试训练数据分离
    :param data:
    :param test_data_pro: (0,1)
    :param seed: 随机数种子
    :return:
    """
    train_data, test_data = data.randomSplit([1.0 - test_data_pro, test_data_pro], seed=seed)
    return train_data, test_data


def get_data_to_info_dict(input_path, col1, col2):
    """
     根据输入数据集读取其路径datatype下json文件，返回一个col1，col2的dict
    :param input_path: 输入数据集的路径
    :param col1: 列1
    :param col2: 列2
    :return: 返回这两列的一个dict，key：col1，value：col2
    """
    df = spark.read.format('json').load(input_path + '/datatype/*')
    row1 = df.select('fieldDefs').collect()[0]
    rows = row1.asDict()['fieldDefs']
    dict_ = {}
    [dict_.update({row[str(col1)]: row[str(col2)]}) for row in rows]
    return dict_
