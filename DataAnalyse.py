
from pyspark.sql import SparkSession
import pandas
from pyspark.sql.functions import *


spark = SparkSession.builder.getOrCreate()

pdDataframe = pandas.read_csv('nuclear_plants_small_dataset.csv')
spDataframe = spark.read.csv('nuclear_plants_small_dataset.csv', header = True, inferSchema = True)

print(pdDataframe)
print(spDataframe)