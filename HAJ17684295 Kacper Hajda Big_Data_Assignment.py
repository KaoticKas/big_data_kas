#Big Data Assignment 1
#HAJ17684295 Kacper Hajda
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
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler


os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.getOrCreate()

spDataframe = spark.read.csv('nuclear_plants_small_dataset.csv', header = True, inferSchema = True)
#pd.set_option("display.max_rows", None, "display.max_columns", None)
spDataframe.select([count(when(isnan(c), c)).alias(c) for c in spDataframe.columns]).show()
 #each columm gets passed through the isNan function, which checks if the column is missing values and counts missing values per column

dfNormal =spDataframe.where(spDataframe.Status =="Normal").drop("Status")
dfAbnormal =spDataframe.where(spDataframe.Status =="Abnormal").drop("Status")
#splits the dataframe into normal and abnormal and drops the status column to only leave features column

def confusion_matrix(predicted_Data):
#this function calculates the confusion matrix statistics to work out sensitivity and Specificity
  truePositive = predicted_Data.where((col("prediction")=="1") & (col("indexedLabel")==1)).count()
  trueNegative = predicted_Data.where((col("prediction")=="0") & (col("indexedLabel")==0)).count()
  falsePositive = predicted_Data.where((col("prediction")=="1") & (col("indexedLabel")==0)).count()
  falseNegative = predicted_Data.where((col("prediction")=="0") & (col("indexedLabel")==1)).count()
  #the confusion matrix takes in the passed data from tested data frame of a model and works out the values based on predicted and actual label
  sensitivity = truePositive/(truePositive+falseNegative) 
  specificity = trueNegative/(trueNegative+falsePositive)
  print("Sensitivity =" + str(sensitivity))
  print("Specificity =" + str(specificity))

normalstats = dfNormal.summary("mean","min", "50%", "max")
#summary provides an insights of the columns and works out the mean, minimum, median(50%) and maximum value found in the columns

modelistpdf = ["mode"]
variances = ['var']
for i in dfNormal.columns:
    modelistpdf.append(dfNormal.groupby(i).count().orderBy("count", ascending=False).first()[0])
    vals = [float(row[i]) for row in dfNormal.select(i).collect()]
    variances.append(np.var(vals))
#I created two lists that will be added to the table to produce a statistics table
#the for loop works out the mode for each feature and adds it to the mode list
#a list of values for a row is collected and then variance is calculated for them and appended to the list 

modepdFrame = pd.DataFrame(modelistpdf)
df_varnorm = pd.DataFrame([variances])

df_varsp = spark.createDataFrame(df_varnorm)

modeListsdf = spark.createDataFrame(modepdFrame.transpose())
statTable = normalstats.union(modeListsdf)

statTable = statTable.union(df_varsp)

print(statTable.toPandas())

#this entire section formats the table of statistics to be one uniform table that will show all the summary statistics
# and mode and varience
print("###############~~~~~~~~~~~~~~~~~~~~~~~Abnormal~~~~~~~~~~~~~~~~~~~~~~~###############")

normalstats2 = dfAbnormal.summary("mean","min", "50%", "max")

modelistpdf2 = ["mode"]
variances2 = ['var']
for i in dfAbnormal.columns:
    modelistpdf2.append(dfAbnormal.groupby(i).count().orderBy("count", ascending=False).first()[0])
    vals2 = [float(row[i]) for row in dfAbnormal.select(i).collect()]
    variances2.append(np.var(vals2))
#Calculates mode and variance for the abnormal group 


modepdFrame2 = pd.DataFrame(modelistpdf2)
df_varnorm2 = pd.DataFrame([variances2])

df_varsp2 = spark.createDataFrame(df_varnorm2)

modeListsdf2 = spark.createDataFrame(modepdFrame2.transpose())# transposes the dataframe to allign with the format of rows and columns of the statistics table
statTable2 = normalstats2.union(modeListsdf2)

statTable2 = statTable2.union(df_varsp2)
print(statTable2.toPandas())
#formatting the dataframe and unifying two dataframes into one to make the stats table for abnormal dataset


col_name = ["Power_range_sensor_1","Power_range_sensor_2","Power_range_sensor_3 ","Power_range_sensor_4","Pressure _sensor_1",
"Pressure _sensor_2","Pressure _sensor_3","Pressure _sensor_4","Vibration_sensor_1","Vibration_sensor_2","Vibration_sensor_3","Vibration_sensor_4"]
#list of column names to produce boxplots

#boxplots
tempDF = spDataframe.toPandas()
for i in col_name:
   sb.boxplot(x='Status',y = i, data=tempDF)
   #plt.show()
   plt.savefig(str(i)+".png")
   plt.clf()
#code that uses seaborn to produce a box plot with the feature and status
#corrolation
print(tempDF.corr())
#non graphical version of the corrolation table
#heatmap
plt.figure(figsize=(16, 6))
heatmap = sb.heatmap(tempDF.corr(), annot=True)
plt.savefig('heatmap-correlation.png', dpi=300, bbox_inches='tight')
plt.clf()
#heatmap of the corrolation table using seaborn


#Classification & Big data analysis 
training_seed = 26
train, test = spDataframe.randomSplit([0.7,0.3], seed= training_seed)
#Creates a random split of data with training getting 70% of the full dataset. The seed allows for testing without the samples changing
print("Training set Rows: "+str(train.count()))
print ("Training Normal: " + str(train.where(col("Status")=="Normal").count()))
print ("Training Abnormal: " + str(train.where(col("Status")=="Abnormal").count()))
print("Testing Set Rows: "+str(test.count()))
print ("Testing Normal: " + str(test.where(col("Status")=="Normal").count()))
print ("Testing Abnormal: " + str(test.where(col("Status")=="Abnormal").count()))

assembler = VectorAssembler( inputCols= col_name, outputCol="features")
# merges multiple columns into a vector column
labelIndexer = StringIndexer(inputCol="Status", outputCol="indexedLabel")
# indexes the status , 1 being Normal 0 Being abnormal and giving it an indexedLabel
#Decision Tree
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features", seed= training_seed)
#uses the function of a decision tree to attempt to classify the data
treePipeline = Pipeline(stages=[assembler, labelIndexer,dt])
#creates a pipline to train the classifier and makes a list of stages to train the model in
model = treePipeline.fit(train) # trains the model based on the train dataset
predictions = model.transform(test)# tests the models classification on the test
predictions.select("prediction", "indexedLabel", "features").show(35)
#shows a table of the classification attempt.

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
#uses an evlauator function to classify how well the classifier identified the labels based on the Incorrectly Classified Samples divided by Classified Sample
print("~~~Decision Tree Classifier~~~")
print("Test Error Rate = %g " % (1.0 - accuracy)) # error rate is 100% - the accuracy of the classifier
confusion_matrix(predictions)
#passes the predicitons frame to the confusion matrix to work the true postives and true negatives.

#Support Vector Model function
svm = LinearSVC(labelCol="indexedLabel", featuresCol="features", predictionCol="prediction")

svmPipeline = Pipeline(stages=[assembler, labelIndexer, svm])# training pipepline
#Fit the model to the train data
svmModel = svmPipeline.fit(train)
predictionsSVM = svmModel.transform(test)
predictionsSVM.select("prediction", "indexedLabel")
print("~~~Support Vector Model~~~")
accuracy2 = evaluator.evaluate(predictionsSVM)
#evaluating performance of the model
print("Test Error Rate = %g " % (1.0 - accuracy2))

confusion_matrix(predictionsSVM)



#artificial neural network
layers = [12, 9, 8, 2]# specifiy the layers of the ANN, 12 features, some intermidate neurons to train and the final classifier Normal or Abnormal
ann = MultilayerPerceptronClassifier(labelCol="indexedLabel", featuresCol="features",layers=layers, seed = training_seed)
annPipeline = Pipeline(stages=[assembler, labelIndexer, ann])# trianing pipeline

annModel = annPipeline.fit(train)
annPredictions = annModel.transform(test)
annPredictions.select("prediction", "indexedLabel")

ANNevaluator = MulticlassClassificationEvaluator(metricName="accuracy")
# Evaluating the performance, by calculating accuracy to work out error rate.
print("~~~Artifical Neural Network Classifier~~~")
accuracy3 = evaluator.evaluate(annPredictions)
print("Test Error Rate = %g " % (1.0 - accuracy3))

confusion_matrix(annPredictions)



#mapreduce (Only got it working on google collab)
def findMaxVal(x):
    yield max(x) 
    #the function finds the highest value in the features column
def findMinVal(x):
    yield min(x)
    #gets min value for each column

def sumMapFinder(x): 
    sumValues = np.array(list(x))
    yield np.sum(sumValues, 0)
    #retruns the sums of the values in a column
    #will be used to work out the mean

def meanOut(sumNumbers, col_val, data_count):
    #function used to output mean values of features
    counter = 0
    #counter to go through each feature
    for value in sumNumbers:
        print(col_val[counter])
        print(value/data_count)#calculates the mean
        counter += 1
        #next column
    


nuclear_largedf = spark.read.csv("nuclear_plants_big_dataset.csv", header=True,inferSchema=True)
nuclear_largedf = nuclear_largedf.drop("Status")
#dropping the status as it is not needed to work out the required values
col_val = nuclear_largedf.schema.names # getting all names of the features
nuclearLRdd = nuclear_largedf.rdd 
# converts the dataframe to an RDD which is a fundemental dataframe
data_count = nuclearLRdd.count() 
# getting count for mean caculation

maximumMap = nuclearLRdd.mapPartitions(findMaxVal)
minimumMap = nuclearLRdd.mapPartitions(findMinVal)
total_Map = nuclearLRdd.mapPartitions(sumMapFinder)
#map paritiion used to run functions on to return values



maxReducer = maximumMap.reduce(lambda x, y: x if (x > y) else y)
#used to get maximum of each feature based on the results of the map.
minReducer = minimumMap.reduce(lambda x, y: x if (x < y) else y)
sumNumbers = total_Map.reduce(lambda x,y: (np.sum([x,y], axis= 0)))
# adds the maps together to get all the values that were in the mpas and its split the result into a 2d array

print("Max values: ")
print(maxReducer)
print("Min values: ")
print(minReducer)
print("Mean values: ")
print(meanOut(sumNumbers, col_val, data_count))
#prints all values calculated