# Import necessary libraries
import torch
from torch_geometric.data import Data, DataLoader  # For graph data and batching
from torch_geometric.nn import GCNConv  # Graph Convolutional Layer
import torch.nn.functional as F
import random  # For generating synthetic data

# Function to generate synthetic dataset
def generate_synthetic_data(num_molecules=100):
    data_list = []
    for i in range(num_molecules):
        # Randomly generate a molecule graph
        num_atoms = random.randint(4, 10)  # Randomly decide the number of atoms in the molecule
        num_bonds = random.randint(num_atoms - 1, num_atoms * 2)  # Random number of bonds
        
        # Node features: One-hot encoding of atom type (e.g., C, O, H)
        x = torch.eye(3)[torch.randint(0, 3, (num_atoms,))]  # 3 atom types: C, O, H
        
        # Edge connections (adjacency relationships between nodes)
        edge_index = torch.randint(0, num_atoms, (2, num_bonds))
        
        # Ensure no self-loops in the graph (e.g., no atom bonded to itself)
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        
        # Edge features: Bond type (e.g., single, double, triple bonds)
        edge_attr = torch.eye(3)[torch.randint(0, 3, (edge_index.shape[1],))]
        
        # Target solubility value (random float for simplicity)
        y = torch.tensor([random.uniform(0, 1)], dtype=torch.float)
        
        # Create a Data object representing the graph
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data_list.append(data)
    return data_list

# Define the GNN model
class SolubilityGNN(torch.nn.Module):
    def __init__(self):
        super(SolubilityGNN, self).__init__()
        self.conv1 = GCNConv(3, 16)  # First GCN layer: 3 input features -> 16 output features
        self.conv2 = GCNConv(16, 32)  # Second GCN layer: 16 input features -> 32 output features
        self.fc = torch.nn.Linear(32, 1)  # Fully connected layer for regression

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply GCN layers with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global mean pooling: Aggregate node features into a single vector
        x = torch.mean(x, dim=0)
        
        # Fully connected layer for solubility prediction
        x = self.fc(x)
        return x

# Generate the dataset
dataset = generate_synthetic_data(100)

# Split dataset into training and testing
train_dataset = dataset[:80]
test_dataset = dataset[80:]

# Create DataLoaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize the model, loss function, and optimizer
model = SolubilityGNN()
criterion = torch.nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer for training

# Training loop
for epoch in range(50):  # Train for 50 epochs
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)  # Forward pass
        loss = criterion(output, batch.y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss:.4f}')

# Evaluate the model on the test dataset
model.eval()
test_loss = 0
for batch in test_loader:
    output = model(batch)  # Forward pass
    loss = criterion(output, batch.y)  # Compute loss
    test_loss += loss.item()
print(f'Test Loss: {test_loss / len(test_loader):.4f}')

