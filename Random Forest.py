import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


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
# model parameters
state = np.random.RandomState(0)
seed = np.random.seed(0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=state)


train_x, test_x = train_test_split(df, test_size=0.33, random_state=0)
train_y = pd.DataFrame()
test_y = pd.DataFrame()
train_y['mark'] = train_x[target_list]
test_y['mark'] = test_x[target_list]
train = train_x[features]
test = test_x[features]

# parameters for Grid Search
parameters = {
    'forest__n_estimators': [int(x) for x in np.linspace(start=10, stop=400, num=100)],
    'forest__max_depth': [int(x) for x in np.linspace(24, 28, num=2)],
}

# Do grid search over k, n_components and C:
numeric_features = features
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

column_trans = ColumnTransformer(
    transformers=[
        ("num",
         numeric_transformer,
         numeric_features)])

# Include the classifier in the main pipeline
pipeline = Pipeline([
    ('features', column_trans),
    ('forest', RandomForestClassifier(class_weight="balanced", random_state=12345))
])

# Perform GridSearch
rf_model = GridSearchCV(pipeline,
                           param_grid=parameters,
                           cv=skf,
                           n_jobs=-1,
                           error_score='raise',
                           scoring='f1_weighted',
                           return_train_score=True,
                           verbose=1)

rf_model.fit(train, train_y)
best = rf_model.best_estimator_
print(rf_model.best_estimator_)
print(rf_model.best_score_)

test_pred = rf_model.predict(test)

# Print the precision and recall, among other metrics
print(classification_report(test_y, test_pred, digits=3))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(
    test_y, test_pred, normalize="true",
    labels=rf_model.classes_),
                       display_labels=rf_model.classes_).plot()
plt.show()