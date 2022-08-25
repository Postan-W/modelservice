from pyspark.ml import Transformer,Estimator
from pyspark.ml.util import JavaMLWriter,MLWritable
from pyspark.ml.param import Param,Params,TypeConverters
from pyspark.sql.functions import col
from pyspark.sql.functions import  udf
from pyspark.sql.functions import  *
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import Pipeline, PipelineModel
# from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.getOrCreate()

from pyspark.ml.util import DefaultParamsReader,DefaultParamsWriter,MLWritable,MLReadable
from pyspark import keyword_only


# 专用于只需要以json形式保存参数的模型的写入
class PythonParamsMLWritable(MLWritable):
    def write(self):
        return PythonParamsMLWriter(self)

class PythonParamsMLWriter(DefaultParamsWriter):
    def __init__(self,instance):
        super(PythonParamsMLWriter,self).__init__(instance)

# 专用于只需要以json形式保存参数的模型的加载
class PythonParamsMLReadable(MLReadable):
    @classmethod
    def read(cls):
        return PythonParamsMLReader(cls)

class PythonParamsMLReader(DefaultParamsReader):
    def __init__(self,cls):
        super(PythonParamsMLReader,self).__init__(cls)





# 需要自己重写的方法，针对pickler或joblib方式保存的方式
# class PythonMLWritable(MLWritable):
#     def write(self):
#         return PythonMLWriter(self)
#
# class PythonMLWriter(DefaultParamsWriter):
#     def __init__(self,instance):
#         super(PythonMLWriter,self).__init__()
#         self.instance = instance
#
#
#     def saveImpl(self, path):
#         pass
#
#     def overwrite(self):
#         return super().overwrite()
#
#     def _handleOverwrite(self, path):
#         super()._handleOverwrite(path)
#
#     def save(self, path):
#         self.write.save(path)
#
#     def session(self, sparkSession):
#         """Sets the Spark Session to use for saving."""
#         self.write.session(sparkSession)
#         return self


#TODO 注意:初始化中继承父类时，使用super().__init__(), 不要使用super(MyClass,self).__init__()，否则可能某些父类属性无法继承
# class PythonEstimator(Estimator,MLWritable):
#     def __init__(self):
#         super().__init__()
#
#     def _fit(self, dataset):
#         """
#         拟合过程
#         :param dataset:
#         :return:
#         """
#         raise NotImplementedError
#
#     def _create_python_model(self,py_model):
#         raise NotImplementedError
#
# class HasFeaturesCol(Params):
#     """
#     Mixin for param featuresCol: features column name.
#     """
#
#     featuresCol = Param(Params._dummy(), "featuresCol", "features column name.", typeConverter=TypeConverters.toString)
#
#     def __init__(self):
#         super(HasFeaturesCol, self).__init__()
#         self._setDefault(featuresCol='features')
#
#     def setFeaturesCol(self, value):
#         """
#         Sets the value of :py:attr:`featuresCol`.
#         """
#         return self._set(featuresCol=value)
#
#     def getFeaturesCol(self):
#         """
#         Gets the value of featuresCol or its default value.
#         """
#         return self.getOrDefault(self.featuresCol)
#
# class HasK(Params):
#     k = Param(Params._dummy(),'k','the constant k',typeConverter=TypeConverters.toFloat)
#
#     @keyword_only
#     def __init__(self):
#         super(HasK,self).__init__()
#         self._setDefault(k=100)
#         kwargs = self._input_kwargs
#         self.setParams(**kwargs)
#
#     @keyword_only
#     def setParams(self,k=100):
#         kwargs = self._input_kwargs
#         return self._set(**kwargs)
#
#     def setK(self,value):
#         return self._set(k=value)
#
#     def getK(self):
#         return self.getOrDefault(self.k)

# class MyKEstimator(PythonEstimator,HasK,PythonParamsMLReadable,PythonParamsMLWritable):
#
#     @keyword_only
#     def __init__(self,k=None):
#         super(MyKEstimator,self).__init__()
#         self._defaultParamMap = {}
#         self._setDefault(k=100)
#         kwargs = self._input_kwargs
#         self.setParams(**kwargs)
#         #避免无法写入问题
#         self.shouldOverwrite = False
#     @keyword_only
#     def setParams(self, k=100):
#         kwargs = self._input_kwargs
#         return self._set(**kwargs)
#
#     def _fit(self, dataset):
#         new_k = dataset.count()
#         self.setK(new_k)
#         return self._create_python_model(new_k)
#
#     def _create_python_model(self,new_k):
#         return MyKModel(k=new_k)

# class MyKModel(Transformer,HasK,PythonParamsMLReadable,PythonParamsMLWritable):
#     def __init__(self,k=None):
#         super(MyKModel,self).__init__()
#         self.setK(k)
#         # 避免无法写入问题
#         self.shouldOverwrite = False
#
#     def _transform(self, dataset):
#
#         k_value = self.getOrDefault(self.k)
#         new_col = str(k_value)
#         from pyspark.sql.functions import lit
#         dataset = dataset.withColumn('new_k',lit(new_col))
#         return dataset





class Hasshopping_col(Params):
    """
    Mixin for param featuresCol: features column name.
    """

    # shopping_col = Param(Params._dummy(), "shopping_col", "shopping_col column name.",typeConverter=TypeConverters.toString)
    shopping_col = Param(Params._dummy(), "shopping_col", "shopping_col column name.",typeConverter=TypeConverters.toString)


    def __init__(self):
        super(Hasshopping_col, self).__init__()
        self._setDefault(shopping_col='shopping_col')

    def setshopping_col(self, value):
        """
        Sets the value of :py:attr:`featuresCol`.
        """
        return self._set(shopping_col=value)

    def getshopping_col(self):
        """
        Gets the value of featuresCol or its default value.
        """
        return self.getOrDefault(self.shopping_col)


    # @keyword_only
    # def __init__(self,shopping_col=None):
    #     super(Hasshopping_col, self).__init__()
    #     # self._setDefault(shopping_col=self.shopping_col)
    #     kwargs = self._input_kwargs
    #     self.setParams(**kwargs)
    #
    # @keyword_only
    # def setParams(self,shopping_col=None):
    #     kwargs = self._input_kwargs
    #     return self._set(**kwargs)
    #
    # def setshopping_col(self, value):
    #     """
    #     Sets the value of :py:attr:`featuresCol`.
    #     """
    #     return self._set(shopping_col=value)
    #
    # def getshopping_col(self):
    #     """
    #     Gets the value of featuresCol or its default value.
    #     """
    #     return self.getOrDefault(self.shopping_col)

class HasitemCol(Params):
    itemCol = Param(Params._dummy(),'itemCol','the constant itemCol',typeConverter=TypeConverters.toString)


    def __init__(self):
        super(HasitemCol, self).__init__()
        self._setDefault(itemCol='itemCol')

    def setitemCol(self, value):
        """
        Sets the value of :py:attr:`featuresCol`.
        """
        return self._set(itemCol=value)

    def getitemCol(self):
        """
        Gets the value of featuresCol or its default value.
        """
        return self.getOrDefault(self.itemCol)



    # @keyword_only
    # def __init__(self):
    #     super(Hasitem_col,self).__init__()
    #     # self._setDefault(itemCol=self.itemCol)
    #     kwargs = self._input_kwargs
    #     self.setParams(**kwargs)
    #
    # @keyword_only
    # def setParams(self,itemCol=itemCol):
    #     kwargs = self._input_kwargs
    #     return self._set(**kwargs)
    #
    # def setitemCol(self,value):
    #     return self._set(itemCol=value)
    #
    # def getitemCol(self):
    #     return self.getOrDefault(self.itemCol)










class MyPipeline(Transformer,Hasshopping_col,HasitemCol,PythonParamsMLReadable,PythonParamsMLWritable):


    @keyword_only
    def __init__(self,shopping_col=None,itemCol=None):
        super(MyPipeline, self).__init__()

        self._setDefault(itemCol='itemCol',shopping_col='shopping_col')
        # self.shopping_col=self.getshopping_col()
        # self.itemCol=self.itemCol
        kwargs = self._input_kwargs
        self.setParams(**kwargs)


        # super(MyPipeline,self).__init__()
        # self._defaultParamMap = {}
        # # self._setDefault(k=100)
        # kwargs = self._input_kwargs
        # self.setParams(**kwargs)
        #避免无法写入问题
        self.shouldOverwrite = False


    @keyword_only
    def setParams(self,itemCol='itemCol',shopping_col='shopping_col'):
        """
        setParams(self, featuresCol="features", labelCol="label", predictionCol="prediction", \
                  probabilityCol="probability", rawPredictionCol="rawPrediction", \
                  maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, \
                  maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity="gini", \
                  seed=None)
        Sets params for the DecisionTreeClassifier.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # def _create_model(self, java_model):
    #     return DecisionTreeClassificationModel(java_model)



    def _transform(self, data):
        shoppings = []
        shopping = []
        # data = data.sort(self.shopping_col)
        data = data.sort(self.getshopping_col())
        data = data.collect()
        current = None
        last = None

        for d in data:
            mm = d[self.getshopping_col()]
            if current == mm or current is None:
                shopping.append(d[self.getitemCol()])
            else:
                shoppings.append((last[self.getshopping_col()], shopping))
                shopping = []
                shopping.append(d[self.getitemCol()])
            last = d
            current = d[self.getshopping_col()]

        shoppings.append((last[self.getshopping_col()], shopping))

        data = spark.createDataFrame(shoppings, [self.getshopping_col(), self.getitemCol()])
        return data
