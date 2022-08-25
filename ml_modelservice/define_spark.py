print("""{"ZERO": 0,"MEAN": 0"}""")
print([{"ZERO": 0,"MEAN": 0}])
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
data = [{"Category": 'Category A', "ID": 1, "Value": 12.40},
        {"Category": 'Category B', "ID": 2, "Value": 30.10},
        {"Category": 'Category C', "ID": 3, "Value": 100.01}
       ]

from ml_server import Serving_Handler

data2 = [{"c_ZERO": 0,"c_MEAN": 0}]
df = spark.createDataFrame(data2)
print(df.schema)
df.show()

