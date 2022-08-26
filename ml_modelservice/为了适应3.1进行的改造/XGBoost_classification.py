# -*- coding: utf-8  -*-
# @Author   :
# @time     : 2019/11/12 09:46:00
# File      : XGBoost_classification.py
# Software  : PyCharm


# from util import keyword_only, JavaClassificationModel
from pyspark import keyword_only
from pyspark.sql import SparkSession
from XGBoost_params import *
# from xgbooster_params import *
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaParams, JavaWrapper
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
# from pyspark.ml.classification import JavaClassificationModel
# from pyspark.ml.common import inherit_doc
from pyspark.ml.param.shared import *
from base import spark
from pyspark.ml.wrapper import JavaPredictionModel
# from pyspark.ml.util import *

# spark = SparkSession.builder \
#     .config('spark.executor.extraClassPath', extra_jar_path) \
#     .config('spark.driver.extraClassPath', extra_jar_path) \
#     .config('spark.executor.extraClassPath', extra_jar_path1) \
#     .config('spark.driver.extraClassPath', extra_jar_path1) \
#     .getOrCreate()


# __all__ = ['XGBoostClassifier', 'XGBoostClassificationModel']


class XGBoostClassifier(JavaEstimator, HasNumRound, HasNumWorkers, HasNThread, HasUseExternalMemory,
                        HasSilent, HasVerbosity, HasMissing, HasTimeoutRequestWorkers, HasCheckpointPath,
                        HasCheckpointInterval, HasSeed, HasEta, HasGamma, HasMaxDepth, HasMaxLeaves,
                        HasMinChildWeight, HasMaxDeltaStep, HasSubsample, HasColSampleByTree, HasColSampleByLevel,
                        HasAlpha, HasTreeMethod, HasGrowPolicy, HasSketchEps,
                        HasScalePosWeight, HasSampleType,
                        HasNormalizeType, HasRateDrop, HasSkipDrop, HasLambdaBias, HasTreeLimit,
                        HasMonotoneConstraints,
                        HasInteractionConstraints, HasObjective, HasObjectiveType, HasBaseScore, HasEvalMetric,
                        HasTrainTestRatio, HasNumEarlyStoppingRounds, HasMaximizeEvaluationMetrics,
                        JavaMLReadable, JavaMLWritable,
                        # SCALA中该XGBoostClassifier类继承自ProbabilisticClassifier的共享参数
                        HasProbabilityCol, HasThresholds, HasRawPredictionCol, HasLabelCol, HasFeaturesCol,
                        HasPredictionCol):

    # lambda_ = Param(Params._dummy(), "", typeConverter=TypeConverters.toFloat)

    @keyword_only
    def __init__(self, numRound=1, numWorkers=1, nthread=1, useExternalMemory=False, silent=0, verbosity=1,
                 missing=None, timeoutRequestWorkers=30 * 60 * 1000, checkpointPath="", checkpointInterval=1, seed=0,
                 eta=0.1, gamma=0, maxDepth=6, maxLeaves=0, minChildWeight=1, maxDeltaStep=0, subsample=0.8,
                 colsampleBytree=1, colSampleByLevel=1, alpha=0, treeMethod='auto', growPolicy='depthwise',
                 sketchEps=0.03, scalePosWeight=1, sampleType='uniform', normalizeType='tree', rateDrop=0.0,
                 skipDrop=0.0, lambdaBias=0, treeLimit=1, objective='binary:logistic', objectiveType='classification',
                 baseScore=0.5, evalMetric='rmse', trainTestRatio=1.0, numEarlyStoppingRounds=0,
                 maximizeEvaluationMetrics=False,
                 # SCALA中该XGBoostClassifier类继承自ProbabilisticClassifier的共享参数
                 featuresCol="features", labelCol="label", predictionCol="prediction", probabilityCol="probability",
                 rawPredictionCol="rawPrediction"):
        """
        __init__(self, numRound=1, numWorkers=1, nthread=1, useExternalMemory=False, silent=0, verbosity=1,
                 missing=None, timeoutRequestWorkers=30 * 60 * 1000, checkpointPath="", checkpointInterval=1, seed=0,
                 eta=0.3, gamma=0, maxDepth=6, maxLeaves=0, minChildWeight=1, maxDeltaStep=0, subsample=1,
                 colSampleByTree=1, colSampleByLevel=1, lambda_=1, alpha=0, treeMethod='auto', growPolicy='depthwise',
                 maxBins=16, sketchEps=0.03, scalePosWeight=1, sampleType='uniform', normalizeType='tree', rateDrop=0.0,
                 skipDrop=0.0, lambdaBias=0, treeLimit=1, objective='reg:squarederror', objectiveType='classification',
                 baseScore=0.5, evalMetric=None, trainTestRatio=1.0, numEarlyStoppingRounds=0,
                 maximizeEvaluationMetrics=None,
                 # SCALA中该XGBoostClassifier类继承自ProbabilisticClassifier的共享参数
                 featuresCol="features", labelCol="label", predictionCol="prediction", probabilityCol="probability",
                 rawPredictionCol="rawPrediction"
                 ):
        """
        # self.numRound = numRound
        super(XGBoostClassifier, self).__init__()
        # super(XGBoostClassifier, self).__init__(numRound=numRound, numWorkers=numWorkers, nthread=nthread,
        #                                         useExternalMemory=useExternalMemory, silent=silent, verbosity=verbosity,
        #                                         missing=missing, timeoutRequestWorkers=timeoutRequestWorkers,
        #                                         checkpointPath=checkpointPath,
        #                                         checkpointInterval=checkpointInterval, seed=seed,
        #                                         eta=eta, gamma=gamma, maxDepth=maxDepth, maxLeaves=maxLeaves,
        #                                         minChildWeight=minChildWeight,maxDeltaStep=maxDeltaStep,
        #                                         subsample=subsample,
        #                                         colSampleByTree=colsampleBytree, colSampleByLevel=colSampleByLevel,
        #                                         lambda_=lambda_, alpha=alpha,
        #                                         treeMethod=treeMethod, growPolicy=growPolicy,
        #                                         maxBins=maxBins, sketchEps=sketchEps, scalePosWeight=scalePosWeight,
        #                                         sampleType=sampleType,
        #                                         normalizeType=normalizeType, rateDrop=rateDrop,
        #                                         skipDrop=skipDrop, lambdaBias=lambdaBias, treeLimit=treeLimit,
        #                                         objective=objective,
        #                                         objectiveType=objectiveType,
        #                                         baseScore=baseScore, evalMetric=evalMetric, trainTestRatio=trainTestRatio,
        #                                         numEarlyStoppingRounds=numEarlyStoppingRounds,
        #                                         maximizeEvaluationMetrics=maximizeEvaluationMetrics,
        #                                         # SCALA中该XGBoostClassifier类继承自ProbabilisticClassifier的共享参数
        #                                         featuresCol=featuresCol, labelCol=labelCol, predictionCol=predictionCol,
        #                                         probabilityCol=probabilityCol,
        #                                         rawPredictionCol=rawPredictionCol)

        # 不用spark的新建objective_api
        self._java_obj = spark.sparkContext._jvm.ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier()
        self._setDefault(numRound=1, numWorkers=1, nthread=1, useExternalMemory=False, silent=0, verbosity=1,
                         missing=None, timeoutRequestWorkers=30 * 60 * 1000, checkpointPath="", checkpointInterval=1,
                         seed=0,
                         eta=0.1, gamma=0, maxDepth=6, maxLeaves=0, minChildWeight=1, maxDeltaStep=0, subsample=0.8,
                         colsampleBytree=1, colsampleBylevel=1, alpha=0, treeMethod='auto',
                         growPolicy='depthwise',
                         sketchEps=0.03, scalePosWeight=1, sampleType='uniform', normalizeType='tree',
                         rateDrop=0.0,
                         skipDrop=0.0, lambdaBias=0, treeLimit=1, objective='binary:logistic',
                         objectiveType='classification',
                         baseScore=0.5, evalMetric='rmse', trainTestRatio=1.0, numEarlyStoppingRounds=0,
                         maximizeEvaluationMetrics=False,
                         # SCALA中该XGBoostClassifier类继承自ProbabilisticClassifier的共享参数
                         featuresCol="features", labelCol="label", predictionCol="prediction",
                         probabilityCol="probability", rawPredictionCol="rawPrediction")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, numRound=1, numWorkers=1, nthread=1, useExternalMemory=False, silent=0, verbosity=1,
                  missing=None, timeoutRequestWorkers=30 * 60 * 1000, checkpointPath="", checkpointInterval=1, seed=0,
                  eta=0.1, gamma=0, maxDepth=6, maxLeaves=0, minChildWeight=1, maxDeltaStep=0, subsample=0.8,
                  colsampleBytree=1, colsampleBylevel=1, alpha=0, treeMethod='auto', growPolicy='depthwise',
                  sketchEps=0.03, scalePosWeight=1, sampleType='uniform', normalizeType='tree',
                  rateDrop=0.0,
                  skipDrop=0.0, lambdaBias=0, treeLimit=1, objective='binary:logistic', objectiveType='classification',
                  baseScore=0.5, evalMetric='rmse', trainTestRatio=1.0, numEarlyStoppingRounds=0,
                  maximizeEvaluationMetrics=False,
                  # SCALA中该XGBoostClassifier类继承自ProbabilisticClassifier的共享参数
                  featuresCol="features", labelCol="label", predictionCol="prediction", probabilityCol="probability",
                  rawPredictionCol="rawPrediction"):
        """
        Sets params for XGBoost.
        If the threshold and thresholds Params are both set, they must be equivalent.
        参数设置
        :param numRound: boosting迭代次数
        :param numWorkers: 用于训练XGBoost模型的任务数（workers）
        :param nthread: 用于执行XGBoost的并行线程数。输入的参数应该<=系统的CPU核心数，若是没有设置算法会检测将其设置为CPU的全部核心数
        :param useExternalMemory: 是否使用外部存储器作为缓存
        :param silent: 设置为0打印运行信息；设置为1静默模式，不打印
        :param verbosity: 打印详细信息。0：silent，1：warning，2：info，3：debug。有时，XGBoost会尝试根据启发更改设置，
        就会展示出警告信息。如果出现异常行为，就会尝试增加信息的详细程度
        :param missing: 缺失值处理，Scala默认float.NaN
        :param timeoutRequestWorkers: 如果numCores不足，该时间请求新的Worker。 如果此值设置为小于或等于0，则将禁用超时
        :param checkpointPath: hdfs文件夹，用于加载和保存检查点。如果设置了checkpointInterval，将会每迭代几次后保存一次检查点
        :param checkpointInterval: checkpoint interval >= 1(-1表示禁用检查点)，例如：10表示模型训练每10次迭代得到一个检查点，
        如果checkpoint interval大于0，必须设置checkpointPath
        :param seed: 随机中子数。设置它可以复现随机数据的结果，也可以用于调整参数
        :param eta: shrinkage参数，用于更新叶子节点权重时，乘以该系数，避免步长过大。参数值越大，越可能无法收敛，一般会
        把学习率eta设置的小一些，迭代次数设置大一点。别名：learning_rate。默认值为0.3
        :param gamma:
        :param maxDepth: 节点分裂所需的最小损失函数下降值。在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点
        :param maxLeaves: 要增加的最大节点数，仅当grow_policy=lossguide时有关
        :param minChildWeight: 决定最小叶子结点样本权重和。XGBoost的这个参数是最小样本权重的和，而GBM参数是最小样本综合。
        这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。但是如果这个值过高，会导致欠拟合。
        这个参数需要使用CV来调整
        :param maxDeltaStep: 允许每个叶节点输出的最大增量步长。该值设置为0，则表示没有约束；设置为正值，则使迭代更加保守。
        通常不需要设置此参数，当样本分类极不均衡时，可能有助于逻辑回归。该值设置为0-10有助于更新迭代
        :param subsample: 这个参数控制对于每棵树，随机采样的比例。减少这个参数的值，算法会更加保守，避免过拟合。但是，
        如果这个值设置的过小，它可能会导致欠拟合。典型值：0.5-1,0.5代表平均采样，防止过拟合
        :param colSampleByTree: 列采样的一组参数
        :param colSampleByLevel: 用来构造每个水平下的列采样率
        :param lambda_: 权重的L2正则化项（和Ridge regression类似）。这个参数是用来控制XGBoost的正则化部分的
        :param alpha: 权重的L1正则化项（和Lasso regression类似）。可以应用在很高维度的情况下，使得算法的速度更快
        :param treeMethod: 可选项：auto、exact、approx、hist.
        auto：小数据集使用贪婪算法，大数据集使用近似算法（单机器学习在旧的使用的是精确算法，当使用近似算法会收到信息）。
        exact：贪婪算法。
        approx：近似算法（分位数和梯度直方图）
        hist：快速直方图的近似贪婪算法（有些性能优化，例如垃圾缓存）
        :param growPolicy: 控制将新节点增加到树的方式。当前仅当tree_method=hist时支持。选项：depthwise、lossguide。
        depthwise：在最靠近根节点处分裂。
        lossguide：在最大损失处分裂。
        :param maxBins: 仅当tree_method=hist时使用，用于存储连续特征的最大不连续箱数。增加此项可提高拆分性，但增加了计算时间
        :param sketchEps: 仅能使用tree_method=approx（近似算法）。这大致转换为0（1/sketch_eps）箱数。与直接选择箱数相比，
        这具有草图正确性的理论保证。通常用户不必调整它，但考虑设置更低的数字以获得更准确的枚举
        :param scalePosWeight: 在各类别样本十分不均衡时，把这个参数设定为一个正值，可以使算法更快收敛。通常可以将其设置
        为负样本的数目与正样本数目的比值
        :param sampleType: 采样算法方式。
        uniform：均匀选择剔除的树。
        weighted：根据权重选择剔除的树。
        :param normalizeType: 归一化算法方式。
        tree：新树的权重与舍弃的树权重一样，新树权重：1/(k+learning_rate),丢弃的树的缩放比例k/(k+learning_rate)。
        forest：新树的权重与舍弃的树权重和一样，新树权重：1/(k+learning_rate),丢弃的树的缩放比例1/(k+learning_rate)。
        :param rateDrop: 丢弃上一轮树的比例
        :param skipDrop: 跳过被舍弃的概率。如果跳过舍弃，新树将采用gbtree方式增加；skip_drop不等于0优先于rate_drop和one_drop
        :param lambdaBias: 线性增压器参数，L2正则化，默认为0（因为偏见不重要，所以不对偏见进行L1调节）
        :param treeLimit: 预测中使用的树木数量； 默认为0（使用所有树）
        :param objective:定义最小化损失函数类型，默认值：reg:squareerror（回归平方损失）。
        常用的类型：
        reg:squarederror： regression with squared loss.（回归平方损失）。
        reg:logistic：逻辑回归。
        binary:logistic：二分类的逻辑回归，输出值为概率。
        binary:logitraw：二分类的逻辑回归，输出逻辑为0/1前一步的分数。
        count:poisson：计数数据的泊松分布，泊松分布的输出均值。默认情况下，在泊松回归中将max_delta_step设置为0.7（用于保证优化）。
        multi:softmax：设置XGBoost使用softmax目标进行多分类，还需设置num_class（类数）。
        multi:softprob：与softmax相同，但输出一个矢量，可以进一步重新整形为矩阵。结果包含属于每个类的每个数据点的预测概率，ndata*nclassdata*nclass。
        rank:pairwise：使用LambdaMART执行列表方式排名，其中规范化折扣积累增益（NDCG）最大化。
        rank:ndcg：使用LambdaMART执行列表方式排名，其中规范化折扣累积增益（NDCG）最大化。
        rank:map：使用LambdaMART执行列表方式排名，其中平均精度（MAP）最大化。
        reg:gamma：使用log-link进行gamma回归。输出是伽玛分布的平均值。它可能是有用的，例如，用于建模保险索赔严重性，或任何可能是伽马分布的结果。
        reg:tweedie：有日志链接的Tweedie回归。它可能是有用的，例如，用于模拟保险中的总损失，或者可能是Tweedie分布的任何结果。
        :param objectiveType: 指定的自定义目标和评估的学习目标类型。
        "regression", "classification"
        :param baseScore: 所有实例的初始化预测分数，全局的偏差值； 只要经历足够多的迭代次数，这个值对最终结果将不会有太大的影响。初始化为0.5
        :param evalMetric: 校验数据所需要的评价指标，不同的目标函数将会有缺省的评价指标（rmse for regression, and error for classification, mean average precision for ranking），用户可以添加多种评价指标，对于Python用户要以list传递参数对给程序，而不是dict，否则后面的参数会覆盖之前。参数选择有 ：
        rmse：均方根误差
        rmsle：均方根对数误差，
        mae：平均绝对误差
        logloss：负对数似然
        error：二分类错误率，#(wrong cases)/#(all cases)。对于预测，评估会将预测值大于0.5的视为正例，其他则视为负例。
        mlogloss：多分类的logloss。
        auc：曲线下面积。
        aucpr：PR曲线下的面积。
        ndcg：归一化折现累积收益。
        map：平均精度。
        gamma-deviance：伽玛回归的剩余偏差。
        :param trainTestRatio: 用于测试的训练比例
        :param numEarlyStoppingRounds: 如果非零，训练将在指定的数字后停止任何评估指标连续增加的百分比
        :param maximizeEvaluationMetrics: 为评估指标定义预期的优化，为最大则为真，否则为最小
        :param featuresCol: 特征列
        :param labelCol: 标签列
        :param predictionCol: 预测列名
        :param probabilityCol: 预测列的类条件概率
        :param rawPredictionCol: 原始预测（也称为置信度）列名称
        :return:
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # lambda为python保留字，无法命名为XGBoost的python参数
    def setLambda(self, value):
        """

        :param lambda_:
        :return:
        """
        self._java_obj.setLambda(lambda_=value)

    def getLambda(self):
        """

        :param sparkContext:
        :return:
        """
        return self.getOrDefault(self.lambda_)

    def _create_model(self, java_model):
        """

        :param java_model:
        :return:
        """
        return XGBoostClassificationModel(java_model)


# class XGBoostClassificationModel(JavaModel, JavaClassificationModel, JavaMLWritable, JavaMLReadable):
class XGBoostClassificationModel(JavaModel, JavaMLWritable, JavaMLReadable):

    def numClasses(self):
        """
        Number of classes (values which the label can take).
        """
        return self._call_java("numClasses")

    @staticmethod
    def load(path):
        """
        模型加载方法
        :param path:  模型的路径
        :return:
        """
        java_clazz = spark.sparkContext._jvm.ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
        return XGBoostClassificationModel(java_clazz.load(path))


# def main():
#     # 断点路径，是根据当前用户所在sparkContext的hadoopConfiguration确定，windows下无法正常使用
#     chkpoint_path = r'C:\Users\YUDAN\Desktop\data1\xgb'
#     input_path = r'C:\Users\YUDAN\Desktop\Data.csv'
#     params_1 = {
#         'numClass': 2,
#         'numRound': 100,
#         # 'num_workers': 1,
#         'missing': 0.0,
#         'eta': 0.1,
#         'maxDepth': 2,
#         'objective': "multi:softprob",
#         'checkpointPath': chkpoint_path
#     }
#
#     # training = spark.read.format("csv").load(input_path)
#     training = spark.read.csv(input_path, header=True)
#
#     # training.show()
#     # xgbc = XGBoostClassifier(**params_)
#     xgbc = XGBoostClassifier(**params_1)
#     xgbc.setFeaturesCol("features") \
#     .setLabelCol("label")\
#
#     # 训练
#     xgb_model = xgbc.fit(training)
#     # 预测
#     prediction = xgb_model.transform(training)
#     #保存模型
#     save_path = r'C:\Users\YUDAN\Desktop\xgb'
#     xgb_model.save(save_path)
#     #加载模型
#     new_xgb_model = XGBoostClassificationModel.load(save_path)
#     #测试新模型
#     new_xgb_model.transform(training)
#
#
# if __name__ == "__main__":
#     main()
