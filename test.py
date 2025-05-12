import torch
import math


def generate_data(batch_size, input_dim):
    """
    Generate synthetic testing data.

    Returns:
        logits: random logit vectors.
        targets: target softmax outputs computed from the logits.
    """
    logits = torch.randn(batch_size, input_dim)
    # Use built-in softmax only for generating targets
    targets = torch.softmax(logits, dim=1)
    return logits, targets


def calculate_metrics(pred, targets):
    """
    Calculate error metrics (MSE, RMSE, MAE, R²) between predictions and targets.

    Returns:
         mse (float): Mean Squared Error.
         rmse (float): Root Mean Squared Error.
         mae (float): Mean Absolute Error.
         r2 (float): R-squared value.
    """
    mse = torch.mean((pred - targets) ** 2)
    rmse = math.sqrt(mse.item())
    mae = torch.mean(torch.abs(pred - targets)).item()
    ss_res = torch.sum((targets - pred) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - ss_res.item() / (ss_tot.item() + 1e-8)
    return mse.item(), rmse, mae, r2


# Set device and load the best saved model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Replace "tuner.pkl" with the correct path to your saved model checkpoint file if needed.
model = torch.load("model.pth", map_location=device,weights_only=False)

model.to(device)
model.eval()

# Generate 1000 random samples for testing.
batch_size = 1000
input_dim = 100 # This should match the input dimension used during training
x, targets = generate_data(batch_size, input_dim)
x, targets = x.to(device), targets.to(device)

# Perform inference.
with torch.no_grad():
    predictions = model(x)

# Calculate and print the error metrics.
mse, rmse, mae, r2 = calculate_metrics(predictions, targets)
print("Test Metrics on 1000 random samples:")
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R-squared (R²): {r2:.6f}")
