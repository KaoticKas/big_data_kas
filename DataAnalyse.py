from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.functions import *
import os 
import sys
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier , LinearSVC , MultilayerPerceptronClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler



os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.getOrCreate()

spDataframe = spark.read.csv('nuclear_plants_small_dataset.csv', header = True, inferSchema = True)
pdDataframe = pd.read_csv('nuclear_plants_small_dataset.csv')

dfNormal =spDataframe.where(spDataframe.Status =="Normal").drop("Status")
dfAbnormal =spDataframe.where(spDataframe.Status =="Abnormal").drop("Status")
#splits the dataframe into normal and abnormal

def confusion_matrix(predicted_Data):
#this function calculates the confusion matrix statistics to work out sensitivity and Specificity
  #total_rows= predicted_Data.count()
  #print("Total test rows ;"+ str(total_rows))
  truePositive = predicted_Data.where((col("prediction")=="1") & (col("indexedLabel")==1)).count()
  trueNegative = predicted_Data.where((col("prediction")=="0") & (col("indexedLabel")==0)).count()
  falsePositive = predicted_Data.where((col("prediction")=="1") & (col("indexedLabel")==0)).count()
  falseNegative = predicted_Data.where((col("prediction")=="0") & (col("indexedLabel")==1)).count()
  sensitivity = truePositive/(truePositive+falseNegative) 
  specificity = trueNegative/(trueNegative+falsePositive)
  print("Sensitivity:" + str(sensitivity))
  print("Specificity:" + str(specificity))

normalstats = dfNormal.summary("mean","min", "50%", "max")

pandasDF = dfNormal.toPandas()


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

print("###############~~~~~~~~~~~~~~~~~~~~~~~Abnormal~~~~~~~~~~~~~~~~~~~~~~~###############")

normalstats2 = dfAbnormal.summary("mean","min", "50%", "max")

pandasDF2 = dfAbnormal.toPandas()


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

col_name = ["Power_range_sensor_1","Power_range_sensor_2","Power_range_sensor_3 ","Power_range_sensor_4","Pressure _sensor_1",
"Pressure _sensor_2","Pressure _sensor_3","Pressure _sensor_4","Vibration_sensor_1","Vibration_sensor_2","Vibration_sensor_3","Vibration_sensor_4"]

#boxplots
tempDF = spDataframe.toPandas()
for i in col_name:
   sb.boxplot(x='Status',y = i, data=tempDF)
   #plt.show()
   plt.savefig(str(i)+".png")
   plt.clf()

#corrolation
print(tempDF.corr())

#heatmap
plt.figure(figsize=(16, 6))
heatmap = sb.heatmap(tempDF.corr(), annot=True)
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
plt.clf()



#Classification & Big data analysis 

train, test = spDataframe.randomSplit([0.7,0.3], 26)
print(train.count())
print(test.count())

assembler = VectorAssembler( inputCols= col_name, outputCol="features")

labelIndexer = StringIndexer(inputCol="Status", outputCol="indexedLabel")


featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures")

#Decision Tree
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

treePipeline = Pipeline(stages=[assembler, labelIndexer, featureIndexer, dt])

model = treePipeline.fit(train)
predictions = model.transform(test)
predictions.select("prediction", "indexedLabel", "features").show(25)


evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("~~~Decision Tree Classifier~~~")
print("Test Error = %g " % (1.0 - accuracy))
confusion_matrix(predictions)

#Support Vector Model
svm = LinearSVC(labelCol="indexedLabel", predictionCol="prediction")

svmPipeline = Pipeline(stages=[assembler, labelIndexer, featureIndexer, svm])
#Fit the model
svmModel = svmPipeline.fit(train)
predictionsSVM = svmModel.transform(test)
predictionsSVM.select("prediction", "indexedLabel")
print("~~~Support Vector Model~~~")
print("Test set accuracy = " + str(evaluator.evaluate(predictionsSVM)))
accuracy2 = evaluator.evaluate(predictionsSVM)
print("Test Error = %g " % (1.0 - accuracy2))

confusion_matrix(predictionsSVM)
#artificial neural network


layers = [12, 9, 8, 2]
ann = MultilayerPerceptronClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",layers=layers, seed=123)
annPipeline = Pipeline(stages=[assembler, labelIndexer, featureIndexer, ann])

annModel = annPipeline.fit(train)
annPredictions = annModel.transform(test)
annPredictions.select("prediction", "indexedLabel")

ANNevaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("~~~Artifical Neural Network Classifier~~~")
print("Test set accuracy = " + str(evaluator.evaluate(annPredictions)))
accuracy3 = evaluator.evaluate(annPredictions)
print("Test Error = %g " % (1.0 - accuracy3))
confusion_matrix(annPredictions)




