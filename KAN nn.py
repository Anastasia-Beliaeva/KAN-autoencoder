#importing all the required libraries
import pandas as pd
import torch
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imodelsx import KANClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

base_path = pathlib.Path(__file__).parent
temp_path = base_path.joinpath('temp_dir')
df = pd.read_excel(base_path.joinpath('df.xlsx'))
df.drop('Unnamed: 0', axis=1, inplace=True)

target_list = 'GGlevel' #['1TLILSCL_TLH6', '1TLILSCL_ILS1', '1TLILSCL_ILS11', '1TLILSCL_TLS6', '1TLILSCL_CLI1_1', '1TLILSCL_CLO1_2', '1TLILSCL_CLF5']
features = ['1TLILSCL_log_introtime', '1TLILSCL_log_fulltime', '1TLILSCL_log_Desktop2_time', '1TLILSCL_log_Explorer3_time',
            '1TLILSCL_log_Browser4_time', '1TLILSCL_log_Search5_time', '1TLILSCL_log_Web6_time',
             '1TLILSCL_log_Download7_time', '1TLILSCL_log_Install8_time',
            '1TLILSCL_log_Install8_DT', '1TLILSCL_log_Install8_explorer', '1TLILSCL_log_tripplanner9_time',
            '1TLILSCL_log_training10_time', '1TLILSCL_log_task1_time', '1TLILSCL_log_task11_time', '1TLILSCL_log_task7_time']
df = df[df != 'undefined']
df.replace('Zone5Request3', 3, inplace=True)
df.replace('Zone5Request5', 5, inplace=True)
df.replace('Zone5Request1', 1, inplace=True)
df.replace('Zone5Request4', 4, inplace=True)
df.replace('Zone5Request2', 2, inplace=True)
df.replace('Zone6Site3', 3, inplace=True)
df.replace('Zone6Site1', 1, inplace=True)
df.replace('Zone6Site4', 4, inplace=True)
df.replace('Zone6Site2', 2, inplace=True)
df = df.apply(pd.to_numeric)
df.dropna(inplace=True)

model = KANClassifier()
X = df[features]
y = df[target_list]
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=12, stratify=y)
#defining hyperparameter ranges
#adjusting hidden layer sizes
hidden_layers = [32, 64]
#different activation regularization values
activations = [0.1, 0.3]
#different entropy regularization values
entropies = [0.1, 0.3]
#different ridge regularization values
ridges = [0.1, 0.3]
#iterating through combinations of hyperparameters including the new dropout rate
for hidden_layer_size in hidden_layers:
    for regularize_activation in activations:
        for regularize_entropy in entropies:
            for regularize_ridge in ridges:
                    kan_model = KANClassifier(hidden_layer_size=hidden_layer_size, device='cpu',
                                          regularize_activation=regularize_activation,
                                          regularize_entropy=regularize_entropy,
                                          regularize_ridge=regularize_ridge)

                    #training the model
                    kan_model.fit(X_train, y_train)
                    #predicting data points using 'kan_model'
                    y_pred = kan_model.predict(X_test)
                    #printing hyperparameter values
                    print(hidden_layer_size, regularize_activation, regularize_entropy, regularize_ridge)
                    #accuracy on test dataset
                    accuracy_test = accuracy_score(y_test, y_pred)
                    print("Accuracy on Test Set:", accuracy_test)

                    #Cohen's Kappa score for the test set
                    kappa_test = cohen_kappa_score(y_test, y_pred)
                    print("Cohen's Kappa Score on Test Set:", kappa_test)

                    #classification report for the test set
                    #print("Classification Report for Test Set:")
                    print(classification_report(y_test, y_pred))

                    #confusion matrix for the test set
                    conf_matrix_test = confusion_matrix(y_test, y_pred)
                    print("Confusion Matrix for Test Set:\n", conf_matrix_test)
                    kan_model.plot()