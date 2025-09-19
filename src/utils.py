from pyspark.sql import SparkSession
from pyspark import SparkContext

def create_spark_session(app_name = "CustomerSegmentation"):
    if SparkContext._active_spark_context is not None:
        SparkContext._active_spark_context.stop()
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()