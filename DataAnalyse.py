from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.functions import *
import os 
import sys
import numpy as np
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.getOrCreate()

spDataframe = spark.read.csv('nuclear_plants_small_dataset.csv', header = True, inferSchema = True)


dfNormal =spDataframe.where(spDataframe.Status =="Normal").drop("Status")
dfAbnormal =spDataframe.where(spDataframe.Status =="Abnormal").drop("Status")
#splits the dataframe into normal and abnormal


normalstats = dfNormal.summary("mean","min", "50%", "max")

pandasDF = dfNormal.toPandas()

#print(pandasDF.loc[:,'Power_range_sensor_1'].mode()) 

modelistpdf = ["mode"]
variances = ['var']
for i in dfNormal.columns:
    modelistpdf.append(dfNormal.groupby(i).count().orderBy("count", ascending=False).first()[0])
    vals = [float(row[i]) for row in dfNormal.select(i).collect()]
    variances.append(np.var(vals))


modepdFrame = pd.DataFrame(modelistpdf)
df_varnorm = pd.DataFrame([variances])

df_varsp = spark.createDataFrame(df_varnorm)

modeListsdf = spark.createDataFrame(modepdFrame.transpose())
statTable = normalstats.union(modeListsdf)

statTable = statTable.union(df_varsp)
print(statTable.toPandas())





print("####################################################################################")
print("###############~~~~~~~~~~~~~~~~~~~~~~~Abnormal~~~~~~~~~~~~~~~~~~~~~~~###############")
print("####################################################################################")
print("####################################################################################")
normalstats2 = dfAbnormal.summary("mean","min", "50%", "max")

pandasDF2 = dfAbnormal.toPandas()

#print(pandasDF.loc[:,'Power_range_sensor_1'].mode()) 

modelistpdf2 = ["mode"]
variances2 = ['var']
for i in dfAbnormal.columns:
    modelistpdf2.append(dfAbnormal.groupby(i).count().orderBy("count", ascending=False).first()[0])
    vals2 = [float(row[i]) for row in dfAbnormal.select(i).collect()]
    variances2.append(np.var(vals2))


modepdFrame2 = pd.DataFrame(modelistpdf2)
df_varnorm2 = pd.DataFrame([variances2])

df_varsp2 = spark.createDataFrame(df_varnorm2)

modeListsdf2 = spark.createDataFrame(modepdFrame2.transpose())
statTable2 = normalstats2.union(modeListsdf2)

statTable2 = statTable2.union(df_varsp2)
print(statTable2.toPandas())