import torch
import torch.nn as nn
import torch.optim as optim
from model import ResnetPretrained

model = ResnetPretrained(3, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


X = torch.rand(
    100,
    3,
    25,
    25,
)

y = torch.randint(0, 3, size=(100,))

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch + 1}/100], Loss: {loss.item()}")

if __name__ == "__main__":
    pass
