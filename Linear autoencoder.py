import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pathlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_path = pathlib.Path(__file__).parent
temp_path = base_path.joinpath('temp_dir')
df = pd.read_excel(base_path.joinpath('df.xlsx'))
df.drop('Unnamed: 0', axis=1, inplace=True)

target_list = 'GGlevel' #['1TLILSCL_TLH6', '1TLILSCL_ILS1', '1TLILSCL_ILS11', '1TLILSCL_TLS6', '1TLILSCL_CLI1_1', '1TLILSCL_CLO1_2', '1TLILSCL_CLF5']
features = ['1TLILSCL_log_introtime', '1TLILSCL_log_fulltime', '1TLILSCL_log_Desktop2_time', '1TLILSCL_log_Explorer3_time',
            '1TLILSCL_log_Browser4_time', '1TLILSCL_log_Search5_time', '1TLILSCL_log_Zone5', '1TLILSCL_log_Web6_time',
            '1TLILSCL_log_Zone6', '1TLILSCL_log_Download7_time', '1TLILSCL_log_Install8_time',
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

# hyperparameters
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-05
len_features = len(features)
hidden1 = 256
hidden2 = 128
hidden3 = 64
hidden4 = 16
len_output = df[target_list].nunique()

# create a dataset class
class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        # self.dataframe = torch.tensor(self.dataframe.to_numpy()).float()
        self.features = self.dataframe[features].values
        self.target = self.dataframe[target_list].values
        self.ids = self.dataframe.index.values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return {'features': self.features[index],
                'targets': self.target[index],
                'index': self.ids}


# Use torch.nn.Module to create models
class Linear(torch.nn.Module):
    def __init__(self, len_features, hidden1, hidden2, len_output):
        super().__init__()
        self.linear1 = torch.nn.Linear(len_features, hidden1)
        self.linear2 = torch.nn.Linear(hidden1, hidden2)
        self.linear3 = torch.nn.Linear(hidden2, hidden3)
        self.linear4 = torch.nn.Linear(hidden3, hidden4)
        self.output = torch.nn.Linear(hidden4, len_output)
        self.double()

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = F.relu(self.linear3(X))
        X = F.relu(self.linear4(X))
        X = self.output(X)
        return X

torch.manual_seed(0)

X_train, X_test = train_test_split(
    PandasDataset(df), test_size=0.33, random_state=0)

train_data_loader = torch.utils.data.DataLoader(X_train,
                                                batch_size=TRAIN_BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=0
                                                )

test_data_loader = torch.utils.data.DataLoader(X_test,
                                              batch_size=TEST_BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=0
                                              )

model = Linear(len_features, hidden1, hidden2, len_output).to(device)
print(model.parameters)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

def loss_fn(outputs, targets):
    loss = torch.nn.CrossEntropyLoss()
    return loss(outputs, targets)

def train_model(n_epochs, training_loader, test_data_loader, model,
                optimizer):
    losses = []
    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        test_loss = 0
        train_f1 = 0
        correct = 0
        for batch_idx, data in enumerate(training_loader):

            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model.forward(data['features'])
            loss = loss_fn(outputs, targets)
            losses.append(loss.detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            train_f1 += multiclass_f1_score(input=torch.argmax(outputs, 1), target=targets, num_classes=4)

        f1 = train_f1 / len(training_loader)
        print('Epoch: {} \tAverage Training F1: {:.6f}'.format(
            epoch,
            f1))
        with torch.no_grad():
            f1_test = 0
            if epoch == 100:
                test_targets = []
                test_outputs = []
                for batch_idx, data in enumerate(test_data_loader):
                    outputs = model.forward(data['features'])
                    targets = data['targets'].to(device, dtype=torch.long)
                    loss = loss_fn(outputs, targets)
                    test_targets.extend(targets)
                    test_outputs.extend(torch.argmax(outputs, 1))
                    test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.item() - test_loss))
                    f1_test += multiclass_f1_score(input=torch.argmax(outputs, 1), target=targets, num_classes=4)
                    cf_matrix = confusion_matrix(test_targets, test_outputs)
                    correct += (torch.argmax(outputs, 1) == targets).sum()/ len(test_data_loader)
                    print(correct/ TEST_BATCH_SIZE)



            train_loss = train_loss / len(training_loader)
            test_loss = test_loss / len(test_data_loader)

            f1_test = f1_test / len(test_data_loader)

            print('Epoch: {} \tAverage Test F1: {:.6f}'.format(
                epoch,
                f1_test))
            print('Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Test Loss: {:.6f}'.format(
                epoch,
                train_loss,
                test_loss
            ))

    return model, test_outputs, test_targets, cf_matrix

trained_model, test_output, test_targets, conf_matrix = \
    train_model(EPOCHS, train_data_loader, test_data_loader, model, optimizer)
