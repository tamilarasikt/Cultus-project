# ==============================
# VAE-BASED ANOMALY DETECTION
# ==============================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

# ------------------------------
# 1. DEVICE CONFIGURATION
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# 2. SET RANDOM SEED
# ------------------------------
np.random.seed(42)
torch.manual_seed(42)

# ------------------------------
# 3. DATA GENERATION
# ------------------------------
n_features = 20
n_train = 1000
n_test_normal = 800
n_test_anomaly = 200

# Normal data
train_data = np.random.normal(0, 1, (n_train, n_features))
test_normal = np.random.normal(0, 1, (n_test_normal, n_features))

# Anomalous data (shifted)
test_anomaly = np.random.normal(5, 1, (n_test_anomaly, n_features))

# Combine test data
X_test = np.vstack([test_normal, test_anomaly])
y_test = np.hstack([
    np.zeros(n_test_normal),
    np.ones(n_test_anomaly)
])

# Convert to tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(X_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(train_tensor),
    batch_size=64,
    shuffle=True
)

# ------------------------------
# 4. VAE MODEL DEFINITION
# ------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# ------------------------------
# 5. BETA-VAE LOSS FUNCTION
# ------------------------------
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.MSELoss(reduction='mean')(recon_x, x)
    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    return recon_loss + beta * kl_loss

# ------------------------------
# 6. MODEL INITIALIZATION
# ------------------------------
latent_dim = 8
beta = 1.0
epochs = 30
lr = 1e-3

model = VAE(n_features, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# ------------------------------
# 7. TRAINING (NORMAL DATA ONLY)
# ------------------------------
model.train()
for epoch in range(epochs):
    total_loss = 0
    for (x,) in train_loader:
        x = x.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar, beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")

# ------------------------------
# 8. ANOMALY SCORING
# ------------------------------
model.eval()
with torch.no_grad():
    X_test_tensor = test_tensor.to(device)
    recon, _, _ = model(X_test_tensor)

    scores = torch.mean(
        (X_test_tensor - recon) ** 2,
        dim=1
    ).cpu().numpy()

# ------------------------------
# 9. METRICS: ROC-AUC & PR-AUC
# ------------------------------
roc_auc = roc_auc_score(y_test, scores)

precision, recall, thresholds = precision_recall_curve(y_test, scores)
pr_auc = auc(recall, precision)

# ------------------------------
# 10. OPTIMIZED F1-SCORE
# ------------------------------
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_f1 = f1_scores[best_idx]
best_threshold = thresholds[best_idx]

# ------------------------------
# 11. FINAL OUTPUT
# ------------------------------
print("\nFINAL EVALUATION RESULTS")
print("--------------------------------")
print(f"ROC-AUC        : {roc_auc:.4f}")
print(f"PR-AUC         : {pr_auc:.4f}")
print(f"Best F1-score  : {best_f1:.4f}")
print(f"Best Threshold : {best_threshold:.4f}")

# ------------------------------
# 12. OPTIONAL VISUALIZATION
# ------------------------------
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (VAE)")
plt.show()
