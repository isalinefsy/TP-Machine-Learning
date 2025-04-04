import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

# Conversion bc_price_evo to 0/1
train_df['bc_price_evo'] = train_df['bc_price_evo'].map({'UP': 0, 'DOWN': 1})
#train_df = train_df[train_df["ab_demand"] != 0.422915]

train_df = train_df.astype(float)
test_df = test_df.astype(float)

# Separate features and target
X = train_df[['date', 'hour', 'bc_price', 'bc_demand', 'ab_price', 'ab_demand', 'transfer']].values
y = train_df['bc_price_evo'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Divide data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define the neural network model
class PricePredictionNN(nn.Module):
    def __init__(self, input_size):
        super(PricePredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# Initialize model, loss function and optimizer
model = PricePredictionNN(input_size=7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).argmax(dim=1)
        val_acc = accuracy_score(y_val.numpy(), val_preds.numpy())
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

# Prediction on test set
X_test = test_df[['date', 'hour', 'bc_price', 'bc_demand', 'ab_price', 'ab_demand', 'transfer']].values
X_test = scaler.transform(X_test)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

test_preds = model(X_test_tensor).argmax(dim=1).numpy()

test_preds_df = pd.DataFrame({
    'bc_price_evo': test_preds
})

# Map predictions back to original labels
test_preds_df['bc_price_evo'] = test_preds_df['bc_price_evo'].map({0: 'DOWN', 1: 'UP'})

# Save predictions to CSV
submission = pd.DataFrame({
    'id': test_df['id'],
    'bc_price_evo': test_preds_df['bc_price_evo']
})
submission.to_csv("submission.csv", index=False)