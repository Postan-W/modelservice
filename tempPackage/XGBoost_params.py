# -*- coding: utf-8  -*-
# @Author   :
# @time     : 2019/11/11 11:07:33
# File      : XGBoost_params.py
# Software  : PyCharm

from pyspark.ml.param import *


# General Params
class HasNumRound(Params):
    """
    The number f rounds for boosting
    """
    numRound = Param(Params._dummy(), "numRound", "The number of rounds for boosting.",
                     typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasNumRound, self).__init__()

    def setNumRound(self, value):
        """
        Set the value of py:attr:'numRound'.
        :param value:
        :return:
        """
        return self._set(numRound=value)

    def getNumRound(self):
        """
        Get the value of numRound or its default value.
        :return:
        """
        return self.getOrDefault(self.numRound)


class HasNumWorkers(Params):
    """
    The number of workers used to train XGBoost model. default: 1
    """
    numWorkers = Param(Params._dummy(), "numWorkers", "number of workers used to run XGBoost.",
                       typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasNumWorkers, self).__init__()

    def setNumWorkers(self, value):
        """
        Set the value of py:attr:'numWorkers'.
        :return:
        """
        return self._set(numWorkers=value)

    def getNumWorkers(self):
        """
        Get the value of numWorkers or its default value.
        :return:
        """
        return self.getOrDefault(self.numWorkers)


class HasNThread(Params):
    """
    The number of threads used by per worker. default 1
    """
    nthread = Param(Params._dummy(), "nthread", "number of threads used by per worker.",
                    typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasNThread, self).__init__()

    def setNThread(self, value):
        """
        Set the value of py:attr:'NThread'.
        :param value:
        :return:
        """
        return self._set(nthread=value)

    def getNThread(self):
        """
        Get the value of NThread or its default value.
        :return:
        """
        return self.getOrDefault(self.nthread)


class HasUseExternalMemory(Params):
    """
    Mixin for param useExternalMemory: whether to use external memory as cache.bool
    """

    useExternalMemory = Param(Params._dummy(), "useExternalMemory", "whether to use external memory as cache.",
                              typeConverter=TypeConverters.toBoolean)

    def __init__(self):
        super(HasUseExternalMemory, self).__init__()

    def setUseExternalMemory(self, value):
        """
        Sets the value of :py:attr:`useExternalMemory`.
        """
        return self._set(useExternalMemory=value)

    def getUseExternalMemory(self):
        """
        Gets the value of useExternalMemory or its default value.
        """
        return self.getOrDefault(self.useExternalMemory)


class HasSilent(Params):
    """
    Mixin for param silent: Deprecated. Please use verbosity instead,0 means printing running messages, 1 means silent mode. (0 or 1).int
    """

    silent = Param(Params._dummy(), "silent", "Deprecated. Please use verbosity instead "
                   + "0 means printing running messages, 1 means silent mode.", typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasSilent, self).__init__()

    def setSilent(self, value):
        """
        Sets the value of :py:attr:`silent`.
        """
        return self._set(silent=value)

    def getSilent(self):
        """
        Gets the value of silent or its default value.
        """
        return self.getOrDefault(self.silent)


class HasVerbosity(Params):
    """
    Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).
   *default: 1
    """
    verbosity = Param(Params._dummy(), "verbosity",
                      "Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info),3 (debug). ",
                      typeConverter=TypeConverters.toInt)

    def __init(self):
        super(HasVerbosity, self).__init__()

    def setVerbosity(self, value):
        """
        Sets the value of :py:attr:`verbosity`.
        :param value:
        :return:
        """
        return self._set(verbosity=value)

    def getVerbosity(self):
        """
        Gets the value of verbosity or its default value.
        :return:
        """
        return self.getOrDefault(self.verbosity)


class HasMissing(Params):
    """
    The value treated as missing. default: Float.NaN
    """

    missing = Param(Params._dummy(), "missing", "the value treated as missing.",
                    typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasMissing, self).__init__()

    def setMissing(self, value):
        """
        Sets the value of :py:attr:`missing`.
        """
        return self._set(missing=value)

    def getMissing(self):
        """
        Gets the value of missing or its default value.
        """
        return self.getOrDefault(self.missing)


class HasTimeoutRequestWorkers(Params):
    """
    the maximum time to wait for the job requesting new workers. default: 30 minutes

    """

    timeoutRequestWorkers = Param(Params._dummy(), "timeoutRequestWorkers", "Mixin for param timeoutRequestWorkers: the"
                                  + "maximum time to request new Workers if numCores are insufficient. Theimeout will be"
                                  + "disabled if this value is set smaller than or equal to 0.",
                                  typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasTimeoutRequestWorkers, self).__init__()

    def setTimeoutRequestWorkers(self, value):
        """
        Sets the value of :py:attr:`timeoutRequestWorkers`.
        """
        return self._set(timeoutRequestWorkers=value)

    def getTimeoutRequestWorkers(self):
        """
        Gets the value of timeoutRequestWorkers or its default value.
        """
        return self.getOrDefault(self.timeoutRequestWorkers)


class HasCheckpointPath(Params):
    """
    The hdfs folder to load and save checkpoint boosters. default: `empty_string`
    """

    checkpointPath = Param(Params._dummy(), "checkpointPath", "the hdfs folder to load and save checkpoints."
                           + "If there are existing checkpoints in checkpoint_path. The job will"
                           + "load the checkpoint with highest version as the starting point for training. If"
                           + "checkpoint_interval is also set, the job will save a checkpoint every a few rounds",
                           typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasCheckpointPath, self).__init__()

    def setCheckpointPath(self, value):
        """
        Set the value of :py:attr:`checkpointPath`.
        """
        return self._set(checkpointPath=value)

    def getCheckpointPath(self):
        """
        Get the value of checkpointPath or its default value.
        """
        return self.getOrDefault(self.checkpointPath)


class HasCheckpointInterval(Params):
    """
    Param for set checkpoint interval (&gt;= 1) or disable checkpoint (-1). E.g. 10 means that
    the trained model will get checkpointed every 10 iterations. Note: `checkpoint_path` must
    also be set if the checkpoint interval is greater than 0.
    """

    checkpointInterval = Param(Params._dummy(), "checkpointInterval", "set checkpoint interval (>= 1) or disable"
                               + "checkpoint (-1). E.g. 10 means that the trained model will get checkpointed every 10"
                               + "iterations. Note: `checkpoint_path` must also be set if the checkpoint interval is greater"
                               + "than 0.",
                               typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasCheckpointInterval, self).__init__()

    def setCheckpointInterval(self, value):
        """
        Set the value of :py:attr:`checkpointInterval`.
        """
        return self._set(checkpointInterval=value)

    def getCheckpointInterval(self):
        """
        Get the value of checkpointInterval or its default value.
        """
        return self.getOrDefault(self.checkpointInterval)


class HasSeed(Params):
    """
    Random seed for the C++ part of XGBoost and train/test splitting.
    """

    seed = Param(Params._dummy(), "seed", "random seed.", typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasSeed, self).__init__()

    def setSeed(self, value):
        """
        Set the value of :py:attr:`seed`.
        """
        return self._set(seed=value)

    def getSeed(self):
        """
        Get the value of seed or its default value.
        """
        return self.getOrDefault(self.seed)


# Booster Params
class HasEta(Params):
    """
    step size shrinkage used in update to prevents overfitting. After each boosting step, we
    can directly get the weights of new features and eta actually shrinks the feature weights
    to make the boosting process more conservative. [default=0.3] range: [0,1]
    """

    eta = Param(Params._dummy(), "eta", "step size shrinkage used in update to prevents overfitting. After each"
                + "boosting step, we can directly get the weights of new features.and eta actually shrinks the feature"
                + "weights to make the boosting process more conservative.",
                typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasEta, self).__init__()

    def setEta(self, value):
        """
        Sets the value of :py:attr:`eta`.
        """
        return self._set(eta=value)

    def getEta(self):
        """
        Gets the value of eta or its default value.
        """
        return self.getOrDefault(self.eta)


class HasGamma(Params):
    """
    minimum loss reduction required to make a further partition on a leaf node of the tree.
    he larger, the more conservative the algorithm will be. [default=0] range: [0,
    Double.MaxValue]
    """

    gamma = Param(Params._dummy(), "gamma", "minimum loss reduction required to make a further partition on a leaf"
                  + "node of the tree. the larger, the more conservative the algorithm will be.",
                  typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasGamma, self).__init__()

    def setGamma(self, value):
        """
        Set the value of :py:attr:`gamma`.
        """
        return self._set(gamma=value)

    def getGamma(self):
        """
        Get the value of gamma or its default value.
        """
        return self.getOrDefault(self.gamma)


class HasMaxDepth(Params):
    """
    maximum depth of a tree, increase this value will make model more complex / likely to be
    overfitting. [default=6] range: [1, Int.MaxValue]
    """

    maxDepth = Param(Params._dummy(), "maxDepth", "maximum depth of a tree, increase this value will make model more"
                     + "complex/likely to be overfitting.",
                     typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasMaxDepth, self).__init__()

    def setMaxDepth(self, value):
        """
        Set the value of :py:attr:`maxDepth`.
        """
        return self._set(maxDepth=value)

    def getMaxDepth(self):
        """
        Get the value of maxDepth or its default value.
        """
        return self.getOrDefault(self.maxDepth)


class HasMaxLeaves(Params):
    """
    Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set.
    """
    maxLeaves = Param(Params._dummy(), "maxLeaves",
                      "Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set.",
                      typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasMaxLeaves, self).__init__()

    def setMaxLeaves(self, value):
        """
        Set the value of :py:attr:`maxLeaves`.
        :param value:
        :return:
        """
        return self._set(maxLeaves=value)

    def getMaxLeaves(self):
        """
        Get the value of maxLeaves or its default value.
        :return:
        """
        return self.getOrDefault(self.maxLeaves)


class HasMinChildWeight(Params):
    """
    minimum sum of instance weight(hessian) needed in a child. If the tree partition step results
    in a leaf node with the sum of instance weight less than min_child_weight, then the building
    process will give up further partitioning. In linear regression mode, this simply corresponds
    to minimum number of instances needed to be in each node. The larger, the more conservative
    he algorithm will be. [default=1] range: [0, Double.MaxValue]
    """

    minChildWeight = Param(Params._dummy(), "minChildWeight", "minimum sum of instance weight(hessian) needed in a"
                           + "child. If the tree partition step results in a leaf node with the sum of instance weight less"
                           + "than min_child_weight, then the building process will give up further partitioning. In linear"
                           + "regression mode, this simply corresponds to minimum number of instances needed to be in each"
                           + "node. The larger, the more conservative the algorithm will be.",
                           typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasMinChildWeight, self).__init__()

    def setMinChildWeight(self, value):
        """
        Set the value of :py:attr:`minChildWeight`.
        """
        return self._set(minChildWeight=value)

    def getMinChildWeight(self):
        """
        Get the value of minChildWeight or its default value.
        """
        return self.getOrDefault(self.minChildWeight)


class HasMaxDeltaStep(Params):
    """
    Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it
    means there is no constraint. If it is set to a positive value, it can help making the update
    step more conservative. Usually this parameter is not needed, but it might help in logistic
    regression when class is extremely imbalanced. Set it to value of 1-10 might help control the
    update. [default=0] range: [0, Double.MaxValue]
    """

    maxDeltaStep = Param(Params._dummy(), "maxDeltaStep", "Maximum delta step we allow each tree's weight estimation"
                         + "to be. If the value is set to 0, it means there is no constraint. If it is set to a positive"
                         + "value, it can help making the update step more conservative. Usually this parameter is not"
                         + "needed, but it might help in logistic regression when class is extremely imbalanced. Set it"
                         + "to value of 1-10 might help control the update", typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasMaxDeltaStep, self).__init__()

    def setMaxDeltaStep(self, value):
        """
        Set the value of :py:attr:`maxDeltaStep`.
        """
        return self._set(maxDeltaStep=value)

    def getMaxDeltaStep(self):
        """
        Get the value of maxDeltaStep or its default value.
        """
        return self.getOrDefault(self.maxDeltaStep)


class HasSubsample(Params):
    """
    subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly
    collected half of the data instances to grow trees and this will prevent overfitting.
    [default=1] range:(0,1]
    """

    subsample = Param(Params._dummy(), "subsample", "subsample ratio of the training instance. Setting it to 0.5 means"
                      + "that XGBoost randomly collected half of the data nstances to grow trees and this will prevent"
                      + "overfitting.",
                      typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasSubsample, self).__init__()

    def setSubsample(self, value):
        """
        Set the value of :py:attr:`subsample`.
        """
        return self._set(subsample=value)

    def getSubsample(self):
        """
        Get the value of subsample or its default value.
        """
        return self.getOrDefault(self.subsample)


class HasColSampleByTree(Params):
    """
    subsample ratio of columns when constructing each tree. [default=1] range: (0,1]
    """

    colsampleBytree = Param(Params._dummy(), "colsampleBytree",
                            "subsample ratio of columns when constructing each tree.",
                            typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasColSampleByTree, self).__init__()

    def setColsampleBytree(self, value):
        """
        Set the value of :py:attr:`colsampleBytree`.
        """
        return self._set(colsampleBytree=value)

    def getColsampleBytree(self):
        """
        Get the value of colsampleBytree or its default value.
        """
        return self.getOrDefault(self.colsampleBytree)


class HasColSampleByLevel(Params):
    """
    subsample ratio of columns for each split, in each level. [default=1] range: (0,1]
    """

    colsampleBylevel = Param(Params._dummy(), "colsampleBylevel",
                             "subsample ratio of columns for each split, in each level.",
                             typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasColSampleByLevel, self).__init__()

    def setColsampleBylevel(self, value):
        """
        Set the value of :py:attr:`colsampleBylevel`.
        """
        return self._set(colsampleBylevel=value)

    def getColsampleBylevel(self):
        """
        Get the value of colsampleBylevel or its default value.
        """
        return self.getOrDefault(self.colsampleBylevel)


class HasLambda_(Params):
    """
    L2 regularization term on weights, increase this value will make model more conservative.
    [default=1]
    """

    lambda_ = Param(Params._dummy(), "lambda_", "L2 regularization term on weights increase this value will make model"
                    + "more conservative", typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasLambda_, self).__init__()

    def setLambda_(self, value):
        """
        Set the value of :py:attr:`lambda_`.
        """
        return self._set(lambda_=value)

    def getLambda_(self):
        """
        Get the value of lambda_ or its default value.
        """
        return self.getOrDefault(self.lambda_)


class HasAlpha(Params):
    """
    L1 regularization term on weights, increase this value will make model more conservative.
    [default=0]
    """

    alpha = Param(Params._dummy(), "alpha", "L1 regularization term on weights, increase this value will make model"
                  + "more conservative.", typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasAlpha, self).__init__()

    def setAlpha(self, value):
        """
        Set the value of :py:attr:`alpha`.
        """
        return self._set(alpha=value)

    def getAlpha(self):
        """
        Get the value of alpha or its default value.
        """
        return self.getOrDefault(self.alpha)


class HasTreeMethod(Params):
    """
    The tree construction algorithm used in XGBoost. options: {'auto', 'exact', 'approx'}
    [default='auto']
    """

    treeMethod = Param(Params._dummy(), "treeMethod", "The tree construction algorithm used in XGBoost,"
                       + "options: {'auto', 'exact', 'approx', 'hist'}", typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasTreeMethod, self).__init__()

    def setTreeMethod(self, value):
        """
        Set the value of :py:attr:`treeMethod`.
        """
        return self._set(treeMethod=value)

    def getTreeMethod(self):
        """
        Get the value of treeMethod or its default value.
        """
        return self.getOrDefault(self.treeMethod)


class HasGrowPolicy(Params):
    """
    growth policy for fast histogram algorithm
    """

    growPolicy = Param(Params._dummy(), "growPolicy", "Controls a way new nodes are added to the tree. Currently"
                       + "supported only if tree_method is set to hist. Choices: depthwise, lossguide. depthwise: split"
                       + "at nodes closest to the root. lossguide: split at nodes with highest loss change.",
                       typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasGrowPolicy, self).__init__()

    def setGrowPolicy(self, value):
        """
        Set the value of :py:attr:`growPolicy`.
        """
        return self._set(growPolicy=value)

    def getGrowPolicy(self):
        """
        Get the value of growPolicy or its default value.
        """
        return self.getOrDefault(self.growPolicy)


class HasMaxBins(Params):
    """
    maximum number of bins in histogram
    """

    maxBins = Param(Params._dummy(), "maxBins", "maximum number of bins in histogram.",
                    typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasMaxBins, self).__init__()

    def setMaxBins(self, value):
        """
        Set the value of :py:attr:`maxBins`.
        """
        return self._set(maxBins=value)

    def getMaxBins(self):
        """
        Get the value of maxBins or its default value.
        """
        return self.getOrDefault(self.maxBins)


class HasSketchEps(Params):
    """
    This is only used for approximate greedy algorithm.
    This roughly translated into O(1 / sketch_eps) number of bins. Compared to directly select
    number of bins, this comes with theoretical guarantee with sketch accuracy.
    [default=0.03] range: (0, 1)
    """

    sketchEps = Param(Params._dummy(), "sketchEps", "This is only used for approximate greedy algorithm. This roughly"
                      + "translated into O(1 / sketch_eps) number of bins. Compared to directly select number of bins, this"
                      + "comes with theoretical guarantee with sketch accuracy.",
                      typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasSketchEps, self).__init__()

    def setSketchEps(self, value):
        """
        Set the value of :py:attr:`sketchEps`.
        """
        return self._set(sketchEps=value)

    def getSketchEps(self):
        """
        Get the value of sketchEps or its default value.
        """
        return self.getOrDefault(self.sketchEps)


class HasScalePosWeight(Params):
    """
    Control the balance of positive and negative weights, useful for unbalanced classes. A typical
    value to consider: sum(negative cases) / sum(positive cases).   [default=1]
    """

    scalePosWeight = Param(Params._dummy(), "scalePosWeight", "Control the balance of positive and negative weights,"
                           + "useful for unbalanced classes. A typical value to consider: sum(negative cases) / sum(positive"
                           + "cases.", typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasScalePosWeight, self).__init__()

    def setScalePosWeight(self, value):
        """
        Set the value of :py:attr:`scalePosWeight`.
        """
        return self._set(scalePosWeight=value)

    def getScalePosWeight(self):
        """
        Get the value of scalePosWeight or its default value.
        """
        return self.getOrDefault(self.scalePosWeight)


class HasSampleType(Params):
    """
    Parameter for Dart booster.
    type of sampling algorithm. "uniform": dropped trees are selected uniformly.
    "weighted": dropped trees are selected in proportion to weight. [default="uniform"]
    """

    sampleType = Param(Params._dummy(), "sampleType", "type of sampling algorithm,options: {'uniform', 'weighted'}",
                       typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasSampleType, self).__init__()

    def setSampleType(self, value):
        """
        Set the value of :py:attr:`sampleType`.
        """
        return self._set(sampleType=value)

    def getSampleType(self):
        """
        Get the value of sampleType or its default value.
        """
        return self.getOrDefault(self.sampleType)


class HasNormalizeType(Params):
    """
    Parameter of Dart booster.
    type of normalization algorithm, options: {'tree', 'forest'}. [default="tree"]
    """

    normalizeType = Param(Params._dummy(), "normalizeType", "type of normalization algorithm,"
                          + "options: {'tree', 'forest'}", typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasNormalizeType, self).__init__()

    def setNormalizeType(self, value):
        """
        Sets the value of :py:attr:`normalizeType`.
        """
        return self._set(normalizeType=value)

    def getNormalizeType(self):
        """
        Gets the value of normalizeType or its default value.
        """
        return self.getOrDefault(self.normalizeType)


class HasRateDrop(Params):
    """
    Parameter of Dart booster.
    dropout rate. [default=0.0] range: [0.0, 1.0]
    """

    rateDrop = Param(Params._dummy(), "rateDrop", "dropout rate.", typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasRateDrop, self).__init__()

    def setRateDrop(self, value):
        """
        Set the value of :py:attr:`rateDrop`.
        """
        return self._set(rateDrop=value)

    def getRateDrop(self):
        """
        Get the value of rateDrop or its default value.
        """
        return self.getOrDefault(self.rateDrop)


class HasSkipDrop(Params):
    """
    Parameter of Dart booster.
    probability of skip dropout. If a dropout is skipped, new trees are added in the same manner
    as gbtree. [default=0.0] range: [0.0, 1.0]
    """

    skipDrop = Param(Params._dummy(), "skipDrop", "probability of skip dropout. If a dropout is skipped, new trees are"
                     + "added in the same manner as gbtree", typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasSkipDrop, self).__init__()

    def setSkipDrop(self, value):
        """
        Set the value of :py:attr:`skipDrop`.
        """
        return self._set(skipDrop=value)

    def getSkipDrop(self):
        """
        Get the value of skipDrop or its default value.
        """
        return self.getOrDefault(self.skipDrop)


class HasLambdaBias(Params):
    """
    Parameter of linear booster
    L2 regularization term on bias, default 0(no L1 reg on bias because it is not important)
    """

    lambdaBias = Param(Params._dummy(), "lambdaBias", "L2 regularization term on bias,default 0 (no L1 reg on bias"
                       + "because it is not important)", typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasLambdaBias, self).__init__()

    def setLambdaBias(self, value):
        """
        Set the value of :py:attr:`lambdaBias`.
        """
        return self._set(lambdaBias=value)

    def getLambdaBias(self):
        """
        Get the value of lambdaBias or its default value.
        """
        return self.getOrDefault(self.lambdaBias)


class HasTreeLimit(Params):
    """

    """
    treeLimit = Param(Params._dummy(), "treeLimit", "number of trees used in the prediction;"
                      + "defaults to 0 (use all trees)", typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasTreeLimit, self).__init__()

    def setTreeLimit(self, value):
        """
        Set the value of :py:attr:`treeLimit`.
        :param value:
        :return:
        """
        return self._set(treeLimit=value)

    def getTreeLimit(self):
        """
        Get the value of treeLimit or its default value.
        :return:
        """
        return self.getOrDefault(self.treeLimit)


class HasMonotoneConstraints(Params):
    """

    """
    monotoneConstraints = Param(Params._dummy(), "monotoneConstraints", "a list in length of number of features, 1"
                                + "indicate monotonic increasing, - 1 means decreasing, 0 means no constraint. If it is"
                                + "shorter than number of features, 0 will be padded",
                                typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasMonotoneConstraints, self).__init__()

    def setMonotoneConstraints(self, value):
        """
        Set the value of :py:attr:`monotoneConstraints`.
        :param value:
        :return:
        """
        return self._set(treeLimit=value)

    def getMonotoneConstraints(self):
        """
        Get the value of monotoneConstraints or its default value.
        :return:
        """
        return self.getOrDefault(self.monotoneConstraints)


class HasInteractionConstraints(Params):
    """

    """
    interactionConstraints = Param(Params._dummy(), "interactionConstraints", "Constraints for interaction representing"
                                   + "permitted interactions. The constraints must be specified in the form of a nest list,"
                                   + "e.g. [[0, 1], [2, 3, 4]],where each inner list is a group of indices of features that"
                                   + "are allowed to interact with each other. See tutorial for more information",
                                   typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasInteractionConstraints, self).__init__()

    def setInteractionConstraints(self, value):
        """
        Set the value of :py:attr:`monotoneConstraints`.
        :param value:
        :return:
        """
        return self._set(interactionConstraints=value)

    def getInteractionConstraints(self):
        """
        Get the value of monotoneConstraints or its default value.
        :return:
        """
        return self.getOrDefault(self.interactionConstraints)


# Learning Task Params
class HasObjective(Params):
    """
    Specify the learning task and the corresponding learning objective.
    options: reg:squarederror, reg:logistic, binary:logistic, binary:logitraw, count:poisson,
    multi:softmax, multi:softprob, rank:pairwise, reg:gamma. default: reg:squarederror
    """

    objective = Param(Params._dummy(), "objective", "objective function used for training, options.",
                      typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasObjective, self).__init__()

    def setObjective(self, value):
        """
        Set the value of :py:attr:`objective`.
        """
        return self._set(objective=value)

    def getObjective(self):
        """
        Get the value of objective or its default value.
        """
        return self.getOrDefault(self.objective)


class HasObjectiveType(Params):
    """
    The learning objective type of the specified custom objective and eval.
    Corresponding type will be assigned if custom objective is defined
    options: regression, classification. default: null
    """

    objectiveType = Param(Params._dummy(), "objectiveType", "objective function used for training, options.",
                          typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasObjectiveType, self).__init__()

    def setObjectiveType(self, value):
        """
        Set the value of :py:attr:`objectiveType`.
        """
        return self._set(objectiveType=value)

    def getObjectiveType(self):
        """
        Get the value of objectiveType or its default value.
        """
        return self.getOrDefault(self.objectiveType)


class HasBaseScore(Params):
    """
    the initial prediction score of all instances, global bias. default=0.5
    """

    baseScore = Param(Params._dummy(), "baseScore", "the initial prediction score of all instances, global bias.",
                      typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasBaseScore, self).__init__()

    def setBaseScore(self, value):
        """
        Set the value of :py:attr:`baseScore`.
        """
        return self._set(baseScore=value)

    def getBaseScore(self):
        """
        Get the value of baseScore or its default value.
        """
        return self.getOrDefault(self.baseScore)


class HasEvalMetric(Params):
    """
    evaluation metrics for validation data, a default metric will be assigned according to
    objective(rmse for regression, and error for classification, mean average precision for
    ranking). options: rmse, mae, logloss, error, merror, mlogloss, auc, aucpr, ndcg, map,
    gamma-deviance
    """

    evalMetric = Param(Params._dummy(), "evalMetric", " evaluation metrics for validation data, a default metric will"
                       + "be assigned according to objective (rmse for regression, and error for classification, mean"
                       + "average precision for ranking),options:",
                       typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasEvalMetric, self).__init__()

    def setEvalMetric(self, value):
        """
        Set the value of :py:attr:`evalMetric`.
        """
        return self._set(evalMetric=value)

    def getEvalMetric(self):
        """
        Get the value of evalMetric or its default value.
        """
        return self.getOrDefault(self.evalMetric)


class HasTrainTestRatio(Params):
    """
    Fraction of training points to use for testing.
    """

    trainTestRatio = Param(Params._dummy(), "trainTestRatio", "fraction of training points to use for testing.",
                           typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasTrainTestRatio, self).__init__()

    def setTrainTestRatio(self, value):
        """
        Set the value of :py:attr:`trainTestRatio`.
        """
        return self._set(trainTestRatio=value)

    def getTrainTestRatio(self):
        """
        Get the value of trainTestRatio or its default value.
        """
        return self.getOrDefault(self.trainTestRatio)


class HasNumEarlyStoppingRounds(Params):
    """
    If non-zero, the training will be stopped after a specified number
    of consecutive increases in any evaluation metric.
    """

    numEarlyStoppingRounds = Param(Params._dummy(), "numEarlyStoppingRounds", "number of rounds of decreasing eval"
                                   + "metric to tolerate before stopping the training.",
                                   typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasNumEarlyStoppingRounds, self).__init__()

    def setNumEarlyStoppingRounds(self, value):
        """
        Set the value of :py:attr:`numEarlyStoppingRounds`.
        """
        return self._set(numEarlyStoppingRounds=value)

    def getNumEarlyStoppingRounds(self):
        """
        Get the value of numEarlyStoppingRounds or its default value.
        """
        return self.getOrDefault(self.numEarlyStoppingRounds)


class HasMaximizeEvaluationMetrics(Params):
    """
    Mixin for param maximizeEvaluationMetrics: max number of iterations (>= 0).
    """

    maximizeEvaluationMetrics = Param(Params._dummy(), "maximizeEvaluationMetrics", "define the expected optimization"
                                      + "to the evaluation metrics, true to maximize otherwise minimize it.",
                                      typeConverter=TypeConverters.toBoolean)

    def __init__(self):
        super(HasMaximizeEvaluationMetrics, self).__init__()

    def setMaximizeEvaluationMetrics(self, value):
        """
        Set the value of :py:attr:`maximizeEvaluationMetrics`.
        """
        return self._set(maximizeEvaluationMetrics=value)

    def getMaximizeEvaluationMetrics(self):
        """
        Get the value of maximizeEvaluationMetrics or its default value.
        """
        return self.getOrDefault(self.maximizeEvaluationMetrics)


class HasWeightCol(Params):
    """

    """
    weightCol = Param(Params._dummy(), "weightCol", "", typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasWeightCol, self).__init__()

    def setWeightCol(self, value):
        """

        """
        return self._set(weightCol=value)

    def getWeightCol(self):
        """

        """
        return self.getOrDefault(self.weightCol)


class HasBaseMarginCol(Params):
    """

    """
    baseMarginCol = Param(Params._dummy(), "baseMarginCol", "", typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasBaseMarginCol, self).__init__()

    def setBaseMarginCol(self, value):
        """

        """
        return self._set(baseMarginCol=value)

    def getBaseMarginCol(self):
        """

        """
        return self.getOrDefault(self.baseMarginCol)


class HasGroupCol(Params):
    """

    """
    groupCol = Param(Params._dummy(), "groupCol", "", typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasGroupCol, self).__init__()

    def setGroupCol(self, value):
        """

        """
        return self._set(groupCol=value)

    def getGroupCol(self):
        """

        """
        return self.getOrDefault(self.groupCol)
