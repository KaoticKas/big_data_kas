from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.functions import *
import os 
import sys
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.getOrCreate()

spDataframe = spark.read.csv('nuclear_plants_small_dataset.csv', header = True, inferSchema = True)
pdDataframe = pd.read_csv('nuclear_plants_small_dataset.csv')

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
#print(statTable.toPandas())



#print("####################################################################################")
#print("###############~~~~~~~~~~~~~~~~~~~~~~~Abnormal~~~~~~~~~~~~~~~~~~~~~~~###############")
#print("####################################################################################")
#print("####################################################################################")
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
#print(statTable2.toPandas())

col_name = ["Power_range_sensor_1","Power_range_sensor_2","Power_range_sensor_3 ","Power_range_sensor_4","Pressure _sensor_1",
"Pressure _sensor_2","Pressure _sensor_3","Pressure _sensor_4","Vibration_sensor_1","Vibration_sensor_2","Vibration_sensor_3","Vibration_sensor_4"]

tempDF = spDataframe.toPandas()
for i in col_name:
   sb.boxplot(x='Status',y = i, data=tempDF)
   #plt.show()
   plt.savefig(str(i)+".png")
   plt.clf()


print(tempDF.corr())


plt.figure(figsize=(16, 6))
heatmap = sb.heatmap(tempDF.corr(), annot=True)

plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
plt.clf()




train, test = spDataframe.randomSplit([0.7,0.3], 26)
print(train.count())
print(test.count())