# Nuclear Reactor Reading data analysis.
#### *This project was completed to forfill the requirments of the big data module*
###### Libraries used; Pandas, PySpark, Seaborn and numpy
In this project, the program reads a csv file that contains records of nuclear reactor sensors with normal and abnormal readings and produces summary insights and graphs like a corrolation matrix and box plots based on the readings. 

The program also trains 3 different machine learning models;
- Decision Tree
- Support Vector Machine
- Artifical Neuron Network

These models were used to evaluate their ability to predict if the reactor was behaving normally or abnormally by calculating their specificity and sensitivity and error rate.

The program also has a section where it used lambda functions and hadoop to calculate data from each feature of the reactor using a dataset that has a lot more records. 

To further improve on this project, A GUI could be provided to provide the statistics in a more readable way rather than using the command line.
