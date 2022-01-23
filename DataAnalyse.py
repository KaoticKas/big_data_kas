
from pyspark.sql import SparkSession
import pandas
from pyspark.sql.functions import *

pdDataframe = pandas.read_csv('')
spDataframe = spark.read.csv('', headers = True, inferSchema = True)














spark = SparkSession.builder.getOrCreate()