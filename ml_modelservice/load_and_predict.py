import json
from base import NoModelAlgorithm, spark
from pyspark.sql.types import StringType
from public import preprocess, rename_code


class Predict(NoModelAlgorithm):
    def __init__(self):
        super().__init__()
        self.model_path = None
        self.prediction_path = None
        # self.features = None

    def set_params(self):
        self.parser.add_argument("--model_path", type=str)
        self.parser.add_argument("--output", type=str)
        self.args = self.parser.parse_args()
        self.after_add_argument()
        print(self.args)

        self.prediction_path = self.args.output
        self.model_path = self.args.model_path
        self.result_load["predictions"] = self.prediction_path
        self.__init_type()
        self.__init_features()

    def __init_type(self):
        """
        初始化类型
        :return:
        """
        json_path = self.concat_path(self.model_path, self.model_key, "type_info", "*")
        print("加载模型type_info，路径为{}".format(json_path))
        try:
            type_info = spark.read.json(json_path)
            type_info_dict = json.loads(type_info.toJSON().collect()[0])
            print("加载的该模型是{0}，该评估类型是{1}".format(self.model_info[type_info_dict["model_type"]],
                                                self.evaluation_info[type_info_dict["evaluation_type"]]))
            self.model_type = type_info_dict["model_type"]
            # self.evaluation_type = type_info_dict["evaluation_type"]  加载中用不到该信息， 屏蔽
        except Exception as e:
            print("*" * 60)
            print("warning")
            print("该模型比较老，请尝试重新运行或者生成该模型，下个版本中将不再支持老的模型")
            print("该模型比较老，请尝试重新运行或者生成该模型，下个版本中将不再支持老的模型")
            print("*" * 60)
            print(e)
            self.model_type = self.model_info.keys()

    def __init_features(self):
        """
        初始化特征列，用于检查数据中是否有该特征列
        :return:
        """
        columns_path = self.concat_path(self.model_path, "columns", "data", "*")
        columns_data = self.read_data(columns_path)
        temp_df = columns_data.toPandas()
        name = "feature_cols"
        # 判断写入数据中的是feature_cols 还是features_col
        if name not in temp_df.values[0]:
            name = "features_col"
        flag = 0
        if "intercept" in temp_df.columns:
            flag = 1
        # feature_cols 需要和模型中columns文件下的对应，写入的时候是调utils.get_columns
        temp_res = temp_df.loc[flag, :] == name
        features_col = temp_res[temp_res.values].index.values
        label_res = temp_df.loc[flag, :] == "label_col"
        label_col = label_res[label_res.values].index.values
        rating_res = temp_df.loc[0,] == "rating_col"
        rating_col = rating_res[rating_res.values].index.values
        # print("模型中的特征列为{}".format(list(features_col)))
        self.features = list(features_col)
        self.label_col = list(label_col)
        self.rating_col = list(rating_col)

    def check_features(self, data):
        """
        检查数据是否有相应的特征列
        :param data: 数据
        :return:
        """
        # 检查特征列是否存在空值
        inner_df = data.select(self.features)
        preprocess(inner_df)
        # 检查训练模型的特征列是否在推理数据集中
        for f in self.features:
            if f not in data.columns:
                raise Exception("训练时的特征 {} 不在该数据中".format(f))

    def try_predict(self, data):
        """
        为了兼容老版本的模型，每个模型try except
        :param data: 数据
        :return:
        """
        for i in self.model_type:
            # noinspection PyBroadException
            try:
                predictions = self.predict(path=self.model_path,
                                           model_type=i,
                                           data=data,
                                           features_col=self.features)
                return predictions
            except:
                continue

    def fit(self, data):
        """
        检测测试数据中是否包含训练数据所有的特征列，并解决
        测试数据中code与训练数据中code不一致（展示名一致）导致模型推理失败的bug
        :param data:
        :return:
        """
        # json_path = self.concat_path(self.model_path, "columns", "datatype", "*")
        # raw_json = self.read_json(json_path)
        # datatype_info = spark.createDataFrame(raw_json.collect()[0]["fieldDefs"])
        # pandas_df = datatype_info.toPandas()
        # temp_df = pandas_df[["code", "displayName"]]
        #
        # self.init_datatype_info()
        # test_data_json = self.datatype_info
        # test_data_name = test_data_json[["code", "displayName"]]
        #
        # data, raw_code_name, raw_test_data_columns, new_code_lst = \
        #     rename_func(data, temp_df, test_data_name, self.label_col[0])

        json_path = self.concat_path(self.model_path, "columns", "datatype", "*")
        raw_json = self.read_json(json_path)
        datatype_info = spark.createDataFrame(raw_json.collect()[0]["fieldDefs"])
        pandas_df = datatype_info.toPandas()
        temp_df = pandas_df[["code", "displayName"]]

        self.init_datatype_info()
        test_data_json = self.datatype_info
        test_data_name = test_data_json[["code", "displayName"]]

        if not self.label_col:
            data, test_code_lst, raw_code_name, raw_test_data_columns = \
                rename_code(data, temp_df, test_data_name, self.rating_col)
        else:
            data, test_code_lst, raw_code_name, raw_test_data_columns = \
                rename_code(data, temp_df, test_data_name, self.rating_col, self.label_col[0])

        if isinstance(self.model_type, int):
            predictions = self.predict(path=self.model_path,
                                       model_type=self.model_type,
                                       data=data,
                                       features_col=self.features)
        else:
            predictions = self.try_predict(data)

        if set(temp_df.code).issubset(set(test_data_name.code)):
            predictions = predictions
        else:
            for i, col in enumerate(raw_code_name):
                if col in predictions.columns:
                    predictions = predictions.withColumnRenamed(col, test_code_lst[i])

        raw_cols = []
        raw_cols.extend(raw_test_data_columns)
        if "probability" in predictions.columns:
            raw_cols.extend(["probability", "prediction"])
        else:
            raw_cols.append("prediction")
        if "prediction_str" in predictions.columns:
            raw_cols.append("prediction_str")

        # for _ in raw_code_name:
        #     predictions = predictions.drop(_)
        # for i, col in enumerate(new_code_lst):
        #     predictions = predictions.withColumnRenamed(col,
        #                                                 raw_test_data_columns[i])
        try:
            predictions = predictions.select(raw_cols)
        except:
            for col, dtype in predictions.dtypes:
                if dtype[:5] == "array":
                    predictions = predictions.withColumn(col,
                                                         predictions[col].cast(StringType()))
        #     # except Exception:
        #     #  目前只有关联规则走这一步逻辑
        #     for col, dtype in predictions.dtypes:
        #         if dtype[:5] == "array":
        #             predictions = predictions.withColumn(col,
        #                                                  predictions[col].cast(StringType()))
        # except Exception:
        #     for col, dtype in predictions.dtypes:
        #         if dtype[:5] == "array":
        #             predictions = predictions.withColumn(col,
        #                                                  predictions[col].cast(StringType()))

        if "prediction_str" in predictions.columns:
            predictions = predictions.drop("prediction")
            predictions = predictions.withColumnRenamed("prediction_str", "prediction")
        return {"predictions": predictions}


def main():
    model = Predict()
    model.set_params()
    data = model.read_input_data()
    result = model.fit(data)
    print("算法执行结束")
    model.write_to_hdfs_nomodel(result)
    print("数据写入hdfs成功")


if __name__ == "__main__":
    main()
