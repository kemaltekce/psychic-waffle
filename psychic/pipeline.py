import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        layer1_output_dim = 10

        self.layer1 = nn.Linear(input_dim, layer1_output_dim)
        self.activation = nn.Tanh()
        self.layer2 = nn.Linear(layer1_output_dim, output_dim)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x


def run():
    torch.manual_seed(456)
    X = torch.randn(10, 1)
    y = X @ torch.tensor([[2.0]]) + torch.tensor(1.0) + torch.rand(10, 1) * 0.1

    model = NeuralNetwork(1, 1)
    learning_rate = 0.01
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    epochs = 100

    for epoch in range(1, epochs + 1):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            print(
                "target, pred: \n", torch.stack([y[:, 0], y_hat[:, 0]], dim=1)
            )
