import torch
import torch.nn as nn
import torch.optim as optim
from custom_state_model import SIModel

# Hyperparameters
input_size = 10
output_size = 6
learning_rate = 0.01
num_epochs = 1000

model = SIModel(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Generate random data
torch.manual_seed(42)  # For reproducibility
inputs = torch.randn(100, input_size)
targets = torch.randn(100, output_size)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training finished!')