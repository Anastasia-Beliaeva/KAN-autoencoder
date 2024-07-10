import torch
import pandas as pd

# Use torch.nn.Module to create models
class AutoEncoder(torch.nn.Module):
    def __init__(self, features: int, hidden: int):
        # Necessary in order to log C++ API usage and other internals
        super().__init__()
        self.encoder = torch.nn.Linear(features, hidden)
        self.decoder = torch.nn.Linear(hidden, features)

    def forward(self, X):
        return self.decoder(self.encoder(X))

    def encode(self, X):
        return self.encoder(X)

# Random data
path ="/Users/aybelyaeva/Desktop/Обработанная_и_прошкалированная_база_данных_с_результатами_обучающихся.xlsx"
df = pd.read_excel(path)
df = df[['Технический индикатор',
       'Технический индикатор.1', 'Технический индикатор.2',
       'Технический индикатор.3', 'Технический индикатор.4',
       'Технический индикатор.5', 'Технический индикатор.6',
       'Технический индикатор.7', 'Технический индикатор.8',
       'Технический индикатор.9', 'Технический индикатор.10',
       'Технический индикатор.11', 'Технический индикатор.12',
       'Технический индикатор.13', 'Технический индикатор.14',
       'Технический индикатор.15', 'Технический индикатор.16',
       'Технический индикатор.17']]
df.drop(0, inplace = True)

features = len(df.columns)
hidden = 10
model = AutoEncoder(features, hidden)
# Pass model.parameters() for increased readability
# Weights of encoder and decoder will be passed
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.Softmax()

# Per-epoch losses are gathered
# Loss is the mean of batch elements, in our case mean of 100 elements
losses = []
for epoch in range(1000):
    reconstructed = model(df)
    loss = loss_fn(reconstructed, df)
    # No need to retain_graph=True as you are not performing multiple passes
    # of backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    losses.append(loss.item())