
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.functions import *


spark = SparkSession.builder.getOrCreate()

pdDataframe = pd.read_csv('nuclear_plants_small_dataset.csv')
spDataframe = spark.read.csv('nuclear_plants_small_dataset.csv', header = True, inferSchema = True)

#print(pdDataframe)
#spDataframe.show()

#For each group, show the following summary statistics for each feature
#in a table: minimum, maximum, mean, median, mode, and variance values. For each group,
#plot the box plot for each feature.

dfNormal =spDataframe.where(spDataframe.Status =="Normal").select(spDataframe.columns[1:13])
dfAbnormal =spDataframe.where(spDataframe.Status =="Abnormal")
# dont forget.toPandas()

normalstats = dfNormal.summary("mean","min", "50%", "max")


#print(dfNormal.loc[:,'Power_range_sensor_1'].mode())

#print(dfNormal.describe(include='all'))

normalstats.show()

modelistpdf = pd.DataFrame([dfNormal.groupby(i).count().orderBy("count", ascending=False).first()[0] for i in dfNormal.columns])

modeListsdf = spark.createDataFrame(modelistpdf.transpose())

modeListsdf.show()
#statTable = normalstats.union(modeListsdf)

#print(statTable.toPandas)


#get rid of the status through normal