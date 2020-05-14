import autogluon as ag
from autogluon import TabularPrediction as task

import bz2
import urllib.request
import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from nni.feature_engineering.gbdt_selector import GBDTSelector

url_zip_train = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2'
urllib.request.urlretrieve(url_zip_train, filename='train.bz2')

f_svm = open('train.svm', 'wt')
with bz2.open('train.bz2', 'rb') as f_zip:
    data = f_zip.read()
    f_svm.write(data.decode('utf-8'))
f_svm.close()

X, y = load_svmlight_file('train.svm')
# X, y = load_svmlight_file(path+'train.csv')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X.shape)
# print(x_train.columns)
path = "../"
# train_data = task.Dataset(file_path=path+"data/train.csv") # error occured if path includes korean
# # train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
# train_data = train_data.head(100)  # subsample 500 s points for faster demo
# print(train_data.head())
#
# label_column = 'SalePrice'
# print("Summary of class variable: \n", train_data[label_column].describe())

dir = path + 'autogluon-test/agModels-predictClass-bin' # specifies folder where to store trained models
predictor = task.fit(train_data=x_train,
                     label=47235,
                     output_directory=dir,
                     eval_metric='mean_squared_error',  # 회귀용, 분류용 메트릭을 구분하여 사용
                     # hyperparameter_tune = True,  # 너무 느림
                     )

results = predictor.fit_summary()
print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon categorized the features as: ", predictor.feature_types)

test_data = task.Dataset(file_path="/s/test.csv")
# y_test = test_data[label_column]  # values to predict
# test_data_nolab = test_data.drop(labels=[label_column], axis=1) # delete label column to prove we're not cheating
# print(test_data_nolab.head())

# predictor = task.load(dir) # unnecessary, just demonstrates how to load previously-trained predictor from file
#
y_pred = predictor.predict(test_data)
print("Predictions:  ", y_pred)
# perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)



