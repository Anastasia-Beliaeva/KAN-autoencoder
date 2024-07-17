import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV


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

# instantiate the model (using the default parameters)
logreg = LogisticRegressionCV(cv=5, random_state=0).fit(df[features], df[target_list])
print(logreg.score(df[features], df[target_list]))