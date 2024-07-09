import torch


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
data = torch.rand(100, 4)
model = AutoEncoder(4, 10)
# Pass model.parameters() for increased readability
# Weights of encoder and decoder will be passed
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Per-epoch losses are gathered
# Loss is the mean of batch elements, in our case mean of 100 elements
losses = []
for epoch in range(1000):
    reconstructed = model(data)
    loss = loss_fn(reconstructed, data)
    # No need to retain_graph=True as you are not performing multiple passes
    # of backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    losses.append(loss.item())