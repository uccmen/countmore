import os
import torch
import torch.nn as nn
import torch.optim as optim

# system inits
os.makedirs("out", exist_ok=True)


# Define the neural network model
class Multiplier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Multiplier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the training loop
def train(model, data_loader, criterion, optimizer, epochs):
    best_loss = None
    for epoch in range(epochs):
        for i, (inputs_batch, targets_batch) in enumerate(data_loader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs_batch)
            loss = criterion(outputs, targets_batch)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
        print("Epoch: {}/{}, Loss: {:.4f}".format(epoch + 1, epochs, loss.item()))

        # Print the loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
            # Test the model on new inputs
            test_inputs = torch.tensor([[6, 6], [7, 7]], dtype=torch.float32)
            test_outputs = model(test_inputs)
            print("Test Results:")
            print(f"Input: {test_inputs[0]}, Output: {test_outputs[0].item()}")
            print(f"Input: {test_inputs[1]}, Output: {test_outputs[1].item()}")
            # save the model to disk if it has improved
            if best_loss is None or loss.item() < best_loss:
                out_path = os.path.join("out", "model.pt")
                print(f"test loss {loss.item()} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = loss.item()

def parse_expression(expression):
    numbers = expression.split("x")
    first_number = int(numbers[0].strip())
    second_number = int(numbers[1].split("=")[0].strip())
    answer = int(numbers[1].split("=")[1].strip())
    return [first_number, second_number], answer

def read_file(file_path):
    inputs = []
    outputs = []
    with open(file_path, "r") as file:
        for line in file:
            input, output = parse_expression(line)
            inputs.append(input)
            outputs.append([output])
    return inputs, outputs

# Define the model, input size, hidden size, and output size
model = Multiplier(2, 128, 1)

print("resuming from existing model in the workdir")
model.load_state_dict(torch.load(os.path.join("out", 'model.pt')))

input_size = 2
hidden_size = 128
output_size = 1

# Define the batch size and number of epochs
batch_size = 50
num_epochs = 60000

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# Define the training data
inputs, outputs = read_file("multiplication-table.txt")
inputs = torch.tensor(inputs, dtype=torch.float32)
labels = torch.tensor(outputs, dtype=torch.float32)

# Create a DataLoader to generate mini-batches
data_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(inputs, labels),
    batch_size=batch_size,
    shuffle=True
)


# Train the model
train(model, data_loader, criterion, optimizer, num_epochs)
