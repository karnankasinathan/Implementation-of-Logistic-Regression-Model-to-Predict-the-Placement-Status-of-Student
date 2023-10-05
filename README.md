# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: karnan k
RegisterNumber: 212222230062

import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()

data1= data.copy()
data1 = data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(y_test,y_pred) 
accuracy 

from sklearn.metrics import confusion_matrix 
confusion = confusion_matrix(y_test,y_pred) 
confusion

from sklearn.metrics import classification_report 
classification_report1 = classification_report(y_test,y_pred) 
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
### 1.Placement data
![e4 1](https://github.com/karnankasinathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787064/44cb701f-754a-444e-82e3-c551fd4ce1ca)

### 2.Salary data
![e4 2](https://github.com/karnankasinathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787064/9436712d-1b0e-4e96-b841-d3b576e7b497)

### 3.Checking the null() function
![e4 3](https://github.com/karnankasinathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787064/2624c723-0124-40e9-8f1c-8d373fd7cef9)

### 4.Data Duplicate
![e4 4](https://github.com/karnankasinathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787064/b7668d21-957d-4d88-ae02-4bb82c759841)

### 5.Print data
![e4 5](https://github.com/karnankasinathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787064/9c15c422-6505-47a2-86dc-1ea61ee8e7d2)

### 6.Data-status of x and y
![e4 6](https://github.com/karnankasinathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787064/6e1cef42-ed5c-4ca4-aa13-5bba2288fc6f)

### 7.y_prediction array
![e4 7](https://github.com/karnankasinathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787064/16464050-36df-43b3-8312-1cb22108509d)

### 8.Accuracy value
![e4 8](https://github.com/karnankasinathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787064/2cc4b319-fbc3-40a2-a5af-89b8aa08466e)

### 9.Confusion array
![e4 9](https://github.com/karnankasinathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787064/5138ed7b-b1e4-4982-b2c6-59b0b065fa75)

### 10.Classification report
![e4 10](https://github.com/karnankasinathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787064/c7945fc0-f938-4904-aa09-5791a14b7716)

### 11.Prediction of LR
 
![e4 11](https://github.com/karnankasinathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787064/715697f2-c66b-4a5f-adec-d8b0e75620a6)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
