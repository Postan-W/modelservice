import argparse
import os
import pandas as pd
import requests
import datetime

from config import hdfs_info
from utils import prob_to_label1

from pyspark.storagelevel import StorageLevel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml import linalg
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import VectorAssembler

# spark = SparkSession.builder \
#     .config("mapreduce.output.fileoutputformat.compress", "false") \
#     .config("spark.broadcast.compress", "false") \
#     .config("spark.sql.parquet.compression.codec", "uncompressed") \
#     .getOrCreate()

spark = SparkSession.builder \
    .config("mapreduce.output.fileoutputformat.compress", "false") \
    .config("spark.broadcast.compress", "false") \
    .config("spark.sql.parquet.compression.codec", "uncompressed") \
    .enableHiveSupport() \
    .getOrCreate()

sc = spark.sparkContext


class Base(object):
    """
    基类，数据中的各个参数
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = None
        self.input_path = None
        self.inputpath_sql = None
        self.output_path = None
        self.exportMode = None
        self.path_sep = "/"  # 路径分隔符
        self.csv_sep = ","
        self.head = True  # csv中是否有表头信息
        self.null_value = "null"  # csv中空值处理
        self.null_able = True
        self.csv_dir_name = "data"  # csv名字
        self.json_dir_name = "datatype"  # json文件名字
        self.vector_name = "vector"
        self.date_name = "date"
        # vector 名字
        self.init_input_output()

    def init_input_output(self):
        self.parser.add_argument("--inputpath",
                                 help="display a inputpath of a given path",
                                 type=str,
                                 default=None)
        self.parser.add_argument("--inputpath_sql",
                                 help="display a inputpath of a given path",
                                 type=str,
                                 default=None)
        self.parser.add_argument("--exportMode",
                                 help="display a mode for write csv",
                                 type=str,
                                 default="overwrite")

    def after_add_argument(self):
        self.input_path = self.args.inputpath
        self.inputpath_sql = self.args.inputpath_sql
        self.exportMode = self.args.exportMode
        # self.output_path = self.args.output

    def get_absolute_data_path(self):
        """
        获得数据的绝对路径
        :return:
        """
        return os.path.join(self.input_path, self.csv_dir_name, "*")

    def get_absolute_json_path(self):
        """
        获得数据类型的绝对路径
        :return:
        """
        return os.path.join(self.input_path, self.json_dir_name, "*")

    def concat_path(self, *subpaths):
        """
        拼接路径
        :param subpaths:
        :return:
        """
        return self.path_sep.join(subpaths)

    def read_data(self, path, schema=None):
        """
        读取数据
        :param: path 路径
        :param: schema: schema信息
        :return: dataframe
        """
        print("the absolute data path is {}".format(path))
        try:
            data = spark.read.csv(path=path, sep=self.csv_sep, header=self.head, schema=schema)
        except Exception as e:
            print(e)
            raise Exception("读取csv数据出错，该csv路径为{}，请检查数据集是否存在。".format(self.input_path))
        return data

    def save_data(self, data, hdfspath, source="HDFS"):
        """
        :param data: 数据
        :param source: 存储方式
        :param hdfspath: 路径
        保存数据：需要将dataframe保存成csv，列数据类型普通的数据类型以及vector
        :return:
        """
        assert isinstance(data, DataFrame)
        for col, dtype in data.dtypes:
            if dtype == self.vector_name:
                data = data.withColumn(col, data[col].cast(StringType()))
        if source == "HDFS":
            try:
                data.write.option("timestampFormat", "yyyy/MM/dd HH:mm:ss ZZ").mode(self.exportMode).csv(
                    hdfspath, header=self.head, emptyValue='')
            except Exception as e:
                print(e)
                raise Exception("写入数据异常，请稍后重试")
        else:
            print("暂不支持的数据写入")

    def save_data_sql(self, data, hdfspath):
        """
        :param data: 数据
        :param hdfspath: 路径
        保存数据：将数据保存为 parquet表中
        :return:
        """
        data.persist()
        data.write.mode(self.exportMode).parquet(hdfspath)
        # 合并多个小文件夹，每个最多128MB
        Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
        FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
        fs = FileSystem.get(sc._jsc.hadoopConfiguration())
        dirSize = fs.getContentSummary(Path(hdfspath)).getLength()
        # print(dirSize)
        fileNum = dirSize / (128 * 1024 * 1024)
        # print(fileNum)
        if fileNum < 1 and self.exportMode == "overwrite":
            # df = sc.sqlContext.read.parquet(hdfspath)
            data.coalesce(1).write.mode("overwrite").parquet(hdfspath)


class JsonImplement(Base):
    """
    josn数据的实现类，实现所有json数据一起处理
    """

    def __init__(self):
        super().__init__()
        self.java_to_python = {
            "java.math.BigDecimal": DoubleType(),
            "java.util.Date": StringType(),  # 时间列比较少，自己做处理
            "java.lang.Boolean": BooleanType(),
            "java.lang.String": StringType(),
            "java.math.BigInteger": LongType()
        }
        self.str_to_spark = {
            "double": DoubleType(),
            "bigint": LongType(),
            "int": IntegerType(),
            "string": StringType(),
            "vector": StringType(),  # 目前先按照string，csv不支持vector
            "boolean": BooleanType(),
            "float": FloatType(),
            "date": StringType()  # 目前date类型按照string读取

        }
        self.python_to_java = {"double": "java.math.BigDecimal",
                               "float": "java.math.BigDecimal",
                               "decimal(38,18)": "java.math.BigDecimal",
                               "date": "java.util.Date",
                               "string": "java.lang.String",
                               "bigint": "java.math.BigInteger",
                               "boolean": "java.lang.Boolean",
                               "int": "java.math.BigInteger",
                               "timestamp[ns]": "java.util.Date",
                               "date32[day]": "java.util.Date",
                               "vector": "java.lang.String"}
        self.str_to_parquet = {"string": "string",
                               "short": "long",
                               "int": "long",
                               "long": "long",
                               "float": "decimal(38,18)",
                               "double": "decimal(38,18)",
                               "decimal": "decimal(38,18)",
                               "date": "string",
                               "timestamp": "string",
                               "boolean": "boolean",
                               "vector": "string"
                               }
        self.spark_type = "sparkType"  # 处理刚上传的数据，经过算子处理的数据json里面都会有这个字段
        self.json_type = {"dataframe_json": 1, "request_json": 2}
        self.datatype_info = None
        self.json_file_name = "Configuration.json"
        self.date_cols = []

    def init_datatype_info(self):
        """
        初始化datatype
        :return:
        """
        path = self.get_absolute_json_path()
        raw_json = self.read_json(path)
        datatype_info = spark.createDataFrame(raw_json.collect()[0]["fieldDefs"]).sort("ord")
        self.datatype_info = datatype_info.toPandas()

    def get_content_from_datatype(self, *col):
        """
        得到datatype中的某些列
        :return:
        """
        temp_lst = list(col)
        info = self.datatype_info[temp_lst]
        return info

    def read_json(self, path):
        """
        spark读取json
        :param: path 路径
        :return:
        """
        print("the absolute json path is {}".format(path))
        try:
            data = spark.read.json(path)
        except Exception as e:
            print(e)
            raise Exception("读取json数据出错，该json路径为{}，请检查json文件是否存在".format(self.input_path))
        return data

    def parse_json(self):
        """
        将spark读进来的json数据提取出需要的数据
        :return: dict
        """
        self.init_datatype_info()
        vector_cols = []
        if self.spark_type in self.datatype_info.columns:
            # 判断vector类型
            vector_cols.extend(self.datatype_info[self.datatype_info[self.spark_type]
                                                  == self.vector_name]["code"])
            # 判断date类型
            self.date_cols.extend(self.datatype_info[self.datatype_info[self.spark_type]
                                                     == self.date_name]["code"])
            datatype_info = self.get_content_from_datatype("code", self.spark_type)
            for key in self.str_to_spark.keys():
                datatype_info = datatype_info.replace(key, self.str_to_spark[key])
            datatype = dict(datatype_info.values)
        else:  # 刚上传的数据
            datatype_info = self.get_content_from_datatype("code", "typeClassName")
            # 判断date类型
            self.date_cols.extend(self.datatype_info[self.datatype_info["typeClassName"]
                                                     == "java.util.Date"]["code"])

            for key in self.java_to_python.keys():  # 替换掉里面java数据类型为spark的数据类型
                datatype_info = datatype_info.replace(key, self.java_to_python[key])
            datatype = dict(datatype_info.values)
        return datatype, vector_cols

    def create_schema(self, json_types):
        """
        构建schema
        :param json_types:
        :return:
        """
        struct = StructType()
        for k, v in json_types.items():
            struct.add(k, v, self.null_able)
        return struct

    def get_display_name(self, codes):
        """
        生成json的时候生成对应的displayname
        :param codes: 需要生成displayname的code
        :return:
        """
        array = []
        datatype = dict(self.get_content_from_datatype("code", "displayName").values)
        # 判断code和displayname，不同，则说明进行中英文转换
        characters = [code for code in datatype.keys() if datatype[code] != code]
        for i in codes:
            if i not in datatype.keys():  # 新生成的列
                candidates = [character for character in characters if character in i]
                # 产生所有的候选集，即旧列的名字被包含在i中
                if candidates:  # 如果该列名不是由某一列生成的，则不需要进行处理
                    candidate = max(candidates, key=len)  # 最长匹配原则
                    i = i.replace(candidate, datatype[candidate])  # 将其中的英文转换为对应的中文
                array.append(i)
            else:  # 原来的列
                array.append(datatype[i])
        return array

    def get_general_field(self, codes, col1, col2):
        """
        生成json的时候生成对应的filed_meaning
        :param codes:需要生成displayname的code
        :param col1: 列名
        :param col2: 列名
        :return:
        """
        datatype = dict(self.get_content_from_datatype(col1, col2).values)
        array = [datatype.get(i) for i in codes]
        return array

    def get_typeclass_name(self, types, default="string"):
        """
        获得对应的typeClassName
        :param types:
        :param default: 找不到就是default
        :return:
        """
        # array = [self.python_to_java.get(i[1], default) for i in types]
        array = []
        for i, j in types:
            # 如果在日期列里面，则特殊处理
            if i in self.date_cols:
                array.append("java.util.Date")
            else:
                if j.startswith("decimal"):
                    array.append("java.math.BigDecimal")
                else:
                    array.append(self.python_to_java.get(j, default))
        return array

    def date_process(self, types):
        """
        对日期类型特殊处理
        :param types: 类型
        """
        array = []
        for code, dtype in types:
            if code in self.date_cols:
                array.append([code, "date"])
            else:
                array.append([code, dtype])
        return array

    def create_json(self, types, datasetid):
        """
        生成保存数据的json
        :param types: dataframe.dtypes
        :param datasetid:
        :return:
        """
        types = self.date_process(types)
        df = pd.DataFrame(types, columns=["code", self.spark_type])
        df = pd.concat([df, pd.DataFrame(
            columns=["name",
                     "displayName",
                     "ord",
                     "typeClassName"])],
                       sort=False)
        df["name"] = df["code"]  # 设置name，一般name和code相同
        df["ord"] = df.index  # ord 顺序
        all_codes = list(df["code"].values)  # 所有的code，包括原来的以及新生成的
        df["displayName"] = self.get_display_name(all_codes)
        df["typeClassName"] = self.get_typeclass_name(types, default="string")
        df = df.transpose()
        df = list(df.to_dict().values())

        param = dict()
        param["dataSetId"] = datasetid
        param["previewFlag"] = "false"
        param["fieldDefs"] = df
        return param

    def save_json(self, param, path, josn_type):
        """
        分为模型的以及非模型的写入方式
        :param param:
        :param path: 路径
        :param josn_type: 1 模型的 2 非模型的
        :return:
        """
        if josn_type not in self.json_type.values():
            raise Exception("暂不支持的json写入方式")
        if josn_type == 1:
            try:
                param.repartition(1).write.json(path)
            except Exception as e:
                print(e)
                raise Exception("json数据写入异常")
        else:
            print("requests_post-----url----请求地址:  ")
            print("requests_post-----para----请求参数:  ")
            print("数据集接口" + path + str(datetime.datetime.now()) + "开始------------------=-----------")
            response = requests.post(url=path, json=param, timeout=3600)
            try:
                print("response------------响应:  ", response)
                res_json = response.json()
                code = res_json.get("code", None)
                msg = res_json.get("msg", None)
                if code != 1000:
                    raise Exception("get bad response when request for json,msg：%s" % msg)
            except Exception:
                raise ("get bad response when request for json,response is %s" % response + '获取接口响应')

    def write_dataframe_json(self, param, full_json_path):
        """
        保存数据的json
        :param param: json
        :param full_json_path:
        :return:
        """
        param_df = spark.createDataFrame([param])
        self.save_json(param_df, full_json_path, 1)

    def write_request_json(self, param, dataset_id=None):
        """
        写入request请求的json数据
        :param param: json数据
        :param dataset_id:
        :return:
        """
        url1 = hdfs_info["url1"]
        self.save_json(param, url1, 2)
        url2 = hdfs_info["url2"]
        param_language = {"dataSetId": dataset_id}
        self.save_json(param_language, url2, 2)


class ModelImplement(Base):
    """
    模型的实现类，实现所有模型相关的读写
    model_type : 0 非模型 1 pipeline_model 2 sklearn_model 3 xgboost_model 4 h2o 5 pmml
    evaluation_type : 0 没有评估类型 1 回归 2 分类 3 聚类 4 Ranking
    """

    def __init__(self, model_type, evaluation_type):
        super().__init__()
        self.model_info = {0: "no_model", 1: "pipeline", 2: "sklearn", 3: "xgboost", 4: "h2o"}
        self.evaluation_info = {0: "no_evaluation", 1: "regression", 2: "classification", 3: "cluster"}
        self.type_info = {}
        self.check_type_info(model_type, evaluation_type)
        self.model_key = "model"  # 输出数据中模型的key

    def check_type_info(self, model_type, evaluation_type):
        """
        判别初始化类型信息是否出错
        :param model_type:
        :param evaluation_type:
        :return:
        """
        if model_type not in self.model_info.keys():
            raise Exception("模型类别错误，模型类别信息为{}".format(self.model_info))
        # else:
        #     print("算法模型分类为{}".format(self.model_info[model_type]).center(100, "="))
        if evaluation_type not in self.evaluation_info.keys():
            raise Exception("评估类别错误，评估类别信息为{}".format(self.evaluation_info))
        # else:
        #     print("算法评估类别为{}".format(self.evaluation_info[evaluation_type]).center(100, "="))
        self.type_info = {"model_type": model_type, "evaluation_type": evaluation_type}

    def pipeline_model_save(self, model, model_path):
        """
        pipeline模型(spark,h2o,xgboost都可以调用该方法进行模型的保存)
        :param model:模型
        :param model_path:路径
        :return:
        """
        full_path = self.concat_path(model_path, self.model_key)
        try:
            model.save(full_path)
        except Exception as e:
            print("模型保存出错")
            raise e

    def sklearn_model_save(self, model, model_path):
        """
        sklearn模型
        :param model:模型
        :param model_path:路径
        :return:
        """
        full_path = self.concat_path(model_path, self.model_key, "metadata")
        # pilkle 文件和 type_info保存在两个文件夹下
        try:
            sc.parallelize([model]).saveAsPickleFile(full_path)
        except Exception as e:
            print("sklearn模型保存出错")
            raise e

    def pipeline_model_load(self, path):
        """
        加载pipeline model
        :param path:
        :return:
        """
        full_path = self.concat_path(path, self.model_key)
        model = PipelineModel.load(full_path)
        return model

    def sklearn_model_load(self, path):
        """
        加载sklearn model
        :param path:
        :return:
        """
        full_path = self.concat_path(path, self.model_key, "metadata")
        model = sc.pickleFile(full_path, 3).collect()[0]
        return model

    def xgb_model_load(self, path):
        """
        加载xgb model
        :param path:
        :return:
        """
        full_path = self.concat_path(path, self.model_key)
        from XGBoost_model_load import load_xgb_model
        xgb_model = load_xgb_model(full_path, "")  # 加载XGBoost模型
        pipeline_model = load_xgb_model(full_path, "PipelineModel")  # 加载处理数据集方法
        return xgb_model, pipeline_model

    def h2o_model_load(self, path):
        """
        加载h2o model
        :param path:
        :return:
        """
        full_path = self.concat_path(path, self.model_key)
        from pysparkling.ml import H2OMOJOSettings, H2OMOJOModel
        settings = H2OMOJOSettings(withDetailedPredictionCol=False)
        model = H2OMOJOModel.createFromMojo(full_path + "/mojo_model", settings)
        return model

    def predict(self, path, model_type, data, features_col):
        """
        接受model的path，然后返回预测的结果
        :param path: 模型路径
        :param model_type: 模型类型
        :param data: 数据
        :param features_col: 特征列，只有在sklearn模型中才会使用到
        :return:
        """
        assert model_type in self.model_info.keys()
        if model_type == 1:
            model = self.pipeline_model_load(path)
            predictions = model.transform(data)
        elif model_type == 2:
            raw_data = data.toPandas()
            f_data = raw_data[features_col]
            x = f_data.values
            model = self.sklearn_model_load(path)
            # if hasattr(model, "predict"):
            #     y = getattr(model, "predict")(x)
            # else:
            #     y = getattr(model, "fit_predict")(x)
            # predictions = raw_data.join(pd.DataFrame(y, columns=["prediction"]))
            # predictions = spark.createDataFrame(predictions)
            # vector = VectorAssembler(inputCols=features_col, outputCol="features")
            # predictions = vector.transform(predictions)
            """
            lightgbm推理时增加概率列
            """
            if hasattr(model, "predict"):
                y = getattr(model, "predict")(x)
                predictions = raw_data.join(pd.DataFrame(y, columns=["prediction"]))
                predictions = spark.createDataFrame(predictions)
                vector = VectorAssembler(inputCols=features_col, outputCol="features")
                predictions = vector.transform(predictions)
                if "probability" not in predictions.columns and hasattr(model, "predict_proba"):
                    # 给lightgbm增加概率列
                    #在raw_data后面增加概率列和预测结果值
                    y = getattr(model, "predict")(x)
                    prob = getattr(model, "predict_proba")(x)
                    prob = [str(i) for i in prob]
                    raw_data['probability'] = prob
                    raw_data['prediction'] = y
                    predictions = raw_data
                    predictions = spark.createDataFrame(predictions)
                    vector = VectorAssembler(inputCols=features_col, outputCol="features")
                    predictions = vector.transform(predictions)
                return predictions

        elif model_type == 3:
            xgb_model, pipeline_model = self.xgb_model_load(path)
            predictions = pipeline_model.transform(data)
            predictions = xgb_model.transform(predictions)
        elif model_type == 4:
            model = self.h2o_model_load(path)
            predictions = model.transform(data)
            # assert isinstance(predictions, DataFrame)
            predictions = predictions.withColumn("probability",
                                                 predictions["prediction"].cast("string"))
            predictions = prob_to_label1(predictions)
        else:
            raise Exception("模型参数错误,模型类别为{}".format(type))
        return predictions


class ModelAlgorithm(ModelImplement, JsonImplement):
    """
    带有模型算法的基类
    """

    def __init__(self, model_type, evaluation_type):
        super().__init__(model_type, evaluation_type)
        self.result_load = {}  # 无需在单独算子中初始化

    def set_params(self):
        raise NotImplementedError

    def read_input_data_csv(self):
        """
        读取数据，将数据根据json转换为对应的dataframe
        :return: dataframe
        """
        path = self.get_absolute_data_path()
        json_type, vector_cols = self.parse_json()
        schema = self.create_schema(json_type)
        raw_data = self.read_data(path, schema)
        if vector_cols:  # 非空说明有vector类型的数据被转成string，尝试转回来
            for col in vector_cols:
                array_udf = udf(lambda x: eval(x), returnType=ArrayType(FloatType()))
                raw_data = raw_data.withColumn(col, array_udf(raw_data[col]))  # string -> array
                vector_udf = udf(lambda vs: linalg.Vectors.dense(vs), linalg.VectorUDT())
                raw_data = raw_data.withColumn(col, vector_udf(raw_data[col]))  # array -> vector
        return raw_data

    def read_input_data(self):
        """
        从parquet表中读取数据
        :return: dataframe
        """
        # spark.sql("show databases").show()
        self.parse_json()
        raw_data = spark.sql(self.inputpath_sql)
        return raw_data

    def write_to_hdfs_model(self, model_result, no_model_result=None):
        """
        写入数据
        :param model_result: 以模型方式输出
        :param no_model_reault: 以非模型方式输出
        :return:
        """
        if no_model_result:
            for k in self.result_load.keys():
                full_data_path = self.concat_path(self.result_load[k], self.csv_dir_name)
                dataset_id = self.result_load[k].split('/')[-1]
                data_json = self.create_json(no_model_result[k].dtypes, datasetid=dataset_id)
                for col, dtype in no_model_result[k].dtypes:
                    if dtype in self.str_to_parquet.keys():
                        no_model_result[k] = no_model_result[k].withColumn(col,
                                                                           no_model_result[k][col].cast(
                                                                               self.str_to_parquet[dtype]))
                self.write_request_json(data_json, dataset_id)
                # self.save_data(no_model_result[k], full_data_path)
                self.save_data_sql(no_model_result[k], full_data_path)

        for k in model_result.keys():
            full_data_path = self.concat_path(self.output_path, k, self.csv_dir_name)
            full_json_path = self.concat_path(self.output_path, k, self.json_dir_name)
            if k != self.model_key:  # 数据
                data_json = self.create_json(model_result[k].dtypes, k)
                # 先生成json，再写入数据，里面有vector类型数据，可以将vector写入sparkTYpe里面
                self.write_dataframe_json(data_json, full_json_path)
                self.save_data(model_result[k], hdfspath=full_data_path)
            else:  # 模型
                if self.type_info["model_type"] == 2:  # 除了sklearn模型，其余的都是pipeline模式保存
                    self.sklearn_model_save(model_result[k], self.output_path)
                else:
                    self.pipeline_model_save(model_result[k], self.output_path)
        # 写入模型信息以及评估类型, 目录在model目录下
        self.write_dataframe_json(self.type_info,
                                  self.concat_path(self.output_path,
                                                   self.model_key,
                                                   "type_info"))


class NoModelAlgorithm(ModelAlgorithm):
    """
    非模型算法的基类
    """

    def __init__(self):
        super().__init__(0, 0)  # 非模型统一类型标识符

    def set_params(self):
        raise NotImplementedError

    def write_to_hdfs_nomodel(self, result):
        """
        非模型写入hdfs
        :param result: 结果
        :return:
        """
        for k in self.result_load.keys():
            full_data_path = self.concat_path(self.result_load[k], self.csv_dir_name)
            dataset_id = self.result_load[k].split('/')[-1]
            data_json = self.create_json(result[k].dtypes, datasetid=dataset_id)
            # print(result[k].dtypes)
            # print(data_json)
            for col, dtype in result[k].dtypes:
                if dtype in self.str_to_parquet.keys():
                    result[k] = result[k].withColumn(col,
                                                     result[k][col].cast(self.str_to_parquet[dtype]))
            # print(result[k].dtypes)
            # self.save_data(result[k], full_data_path)
            self.save_data_sql(result[k], full_data_path)
            self.write_request_json(data_json, dataset_id)


if __name__ == '__main__':
    """
    input_path 必须有这样一个参数
    result_load 不需要在每个算子中声明
    vector 不需要保存成string，base中做处理，并且能够加载进来
    如果是模型，则结果中必须由一个self.model_key
    每个算子必须初始化type_info  ModelAlgorithm(1, 2)
    get_label_class_num 调用方式发生改变
    get_displayname 可以通过datatype_info,然后通过def get_content_from_datatype获得，最后加一部 dict()
    self.output = None 模型类不需要在每个算子中单独声明
    inputpath 
    """
    ai = ModelAlgorithm(1, 2)
    print(ai.path_sep)
