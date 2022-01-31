
from pyspark.sql import SparkSession
import pandas
from pyspark.sql.functions import *


spark = SparkSession.builder.getOrCreate()

pdDataframe = pandas.read_csv('nuclear_plants_small_dataset.csv')
spDataframe = spark.read.csv('nuclear_plants_small_dataset.csv', header = True, inferSchema = True)

#print(pdDataframe)
#spDataframe.show()

#For each group, show the following summary statistics for each feature
#in a table: minimum, maximum, mean, median, mode, and variance values. For each group,
#plot the box plot for each feature.

dfNormal =spDataframe.where(spDataframe.Status =="Normal").toPandas()
dfAbnormal =spDataframe.where(spDataframe.Status =="Abnormal").toPandas()


#dfNormal.summary("mean","min", "50%", "max").show()

#normalMode= dfNormal.mode()[1:12]


print(dfNormal.loc[:,'Power_range_sensor_1'].mode())

print(dfNormal.describe(include='all'))