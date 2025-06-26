import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

# Pad naar Alpine_Track_1.csv
CSV_PATH = "combined_data_cleaned.csv"
MODEL_PATH = "alpine_expert_model.pt"
SCALER_PATH = "alpine_expert_scaler.save"

# Laad data
print(f"Laad data uit {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Kolomnamen bepalen (pas aan als nodig)
feature_cols = [
    "SPEED", "TRACK_POSITION", "ANGLE_TO_TRACK_AXIS"
] + [f"TRACK_EDGE_{i}" for i in range(19)]
target_cols = ["ACCELERATION", "BRAKE", "STEERING"]

# Check of alle kolommen aanwezig zijn
for col in feature_cols + target_cols:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' ontbreekt in de data!")

X = df[feature_cols].values
Y = df[target_cols].values

# Clip targets naar juiste bereik
Y[:,0] = np.clip(Y[:,0], 0, 1)  # ACCELERATION
Y[:,1] = np.clip(Y[:,1], 0, 1)  # BRAKE
Y[:,2] = np.clip(Y[:,2], -1, 1) # STEERING

# Normaliseer features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)

# PyTorch Dataset
class TorcsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = TorcsDataset(X_scaled, Y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

# PyTorch Model
class TorcsNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

input_dim = X.shape[1]
model = TorcsNet(input_dim)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
print("Start training...")
epochs = 500
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, Y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        # Outputs: accel/brake [0,1], steering [-1,1]
        outputs_clipped = outputs.clone()
        outputs_clipped[:,0] = torch.clamp(outputs[:,0], 0, 1)
        outputs_clipped[:,1] = torch.clamp(outputs[:,1], 0, 1)
        outputs_clipped[:,2] = torch.clamp(outputs[:,2], -1, 1)
        loss = loss_fn(outputs_clipped, Y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    epoch_loss = running_loss / len(dataset)
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# Opslaan
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model opgeslagen als {MODEL_PATH}")
print(f"Scaler opgeslagen als {SCALER_PATH}")