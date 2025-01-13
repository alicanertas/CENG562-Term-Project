import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from rdkit import Chem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This function converts SMILES codes into molecular structure
def smiles_to_graph(smiles, target):
    #target = [10**target] #it might be used later to convert logarithmic inputs to normal inputs
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None 

    # AddHs function adds H atoms and connects to other atoms according to the maximum numbers of bond those atoms may have
    mol = Chem.AddHs(mol)

    # Node features: Atom type 
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    node_features = torch.eye(100)[torch.tensor(atom_types)]  # Support up to atomic number 99

    # Edge features: Bond type 
    edge_indices = []
    edge_features = []
    bond_types = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        bond_type = bond.GetBondType()
        edge_features.append(bond_types[bond_type])
        edge_features.append(bond_types[bond_type])

    edge_index = torch.tensor(edge_indices).t().contiguous()
    edge_attr = torch.eye(4)[torch.tensor(edge_features)]  # 4 bond types include single, double, triple, and aromatic bonds.

    # the target property is solubility
    y = torch.tensor([target], dtype=torch.float)
    
    
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)

file_path = "CENG562_TermProject/data/delaney-processed.csv"

# this section tries to read the contents and group the information received from the input file
try:
    df = pd.read_csv(file_path)
    df_columns = df.columns.tolist()
    df_head = df.head()
    print("Dataset Columns:", df_columns)
    print("First Rows of Dataset:\n", df_head)
except Exception as e:
    print("Error reading the dataset:", e)
    exit()

# Graph Construction from dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    graphs = []
    for _, row in df.iterrows():
        graph = smiles_to_graph(row['smiles'], row['measured log solubility in mols per litre'])
        if graph is not None:
            graphs.append(graph)
    return graphs

# This class defines the structure of the GNN model
class SolubilityGNN(torch.nn.Module):
    def __init__(self):
        super(SolubilityGNN, self).__init__()
        self.conv1 = GCNConv(100, 64)  # Input: 100 features -> Output: 64 features
        self.conv2 = GCNConv(64, 32)  # Input: 64 features -> Output: 32 features
        self.fc = torch.nn.Linear(32, 1)  # The final layer resulting a single scalar

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    
    dataset = load_dataset(file_path)

    # Dataset is splitted into two parts, 80% for training and 20% for testing
    train_dataset = dataset[:int(0.8 * len(dataset))]
    test_dataset = dataset[int(0.2 * len(dataset)):]

    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=1)  

    # the loss function is RMSE, and the optimization algorithm is Adam optimizer with a learning rate of 0.01 
    model = SolubilityGNN()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    model.eval()
    test_loss = 0
    all_predictions = []
    all_true_values = []
    all_component_names = []

    # This loop returns the prediction results with losses.
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch)
            
            all_component_names.extend(batch.smiles)  

            output = output.view(-1) 
            batch.y = batch.y.view(-1)  

            # Loss calculation
            loss = criterion(output, batch.y)
            test_loss += loss.item()

            #  It is to collect predictions and actual values
            all_predictions.extend(output.cpu().numpy())
            all_true_values.extend(batch.y.cpu().numpy())

    # After loop ends, theese lines calculate RMSE
    print(f"Predictions length: {len(all_predictions)}")
    print(f"True values length: {len(all_true_values)}")

    if len(all_predictions) == len(all_true_values):
        rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_true_values)) ** 2))
        aard = np.mean(np.abs((np.array(all_predictions) - np.array(all_true_values)) / np.array(all_true_values)))
        print(f"Test Loss (Normal scale): {test_loss / len(test_loader):.4f}")
        print(f"Test RMSE (Normal scale): {rmse:.4f}")
        print(f"Test AARD (fraction): {aard:.4f}")

        for smile, prediction, true_value in zip(all_component_names, all_predictions, all_true_values):
            print(f"SMILES: {smile} | Predicted: {prediction:.4f} | Actual: {true_value:.4f}")

        # This section provides the codes to plot the results
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 4))
        plt.scatter(all_true_values, all_predictions, alpha=0.7, edgecolors='k', label='Data points')

        # Here the y=x line is plotted to compare the prediction results with the expectations
        min_val = min(min(all_true_values), min(all_predictions))
        max_val = max(max(all_true_values), max(all_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')

        plt.xlabel("Experimental Log Solubility (mol/L)")
        plt.ylabel("Predicted Log Solubility (mol/L)")
        plt.title("Predicted vs Experimental Log Solubility")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
    else:
        print("Error: Mismatch in the number of predictions and true values.")
