from pathlib import Path

import pyarrow
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import math
from ray import tune
import os

from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import stopper
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

from models import SoftmaxApprox


def generate_data(batch_size, input_dim):
    """
    Generate synthetic training data.

    Returns:
        logits: random logit vectors.
        targets: target softmax outputs computed from the logits.
    """

    logits = torch.randn((batch_size, input_dim))
    # Use built-in softmax only for data generation
    targets = torch.softmax(logits, dim=-1)
    return logits, targets


def calculate_metrics(pred, targets):
    """
    Calculate additional metrics (RMSE, MAE, R^2) between predictions and targets.

    Returns:
         rmse (float): Root Mean Squared Error.
         mae (float): Mean Absolute Error.
         r2 (float): R-squared value.
    """
    # Mean Squared Error (MSE)
    mse = torch.mean((pred - targets) ** 2)
    # Root Mean Squared Error (RMSE)
    rmse = math.sqrt(mse.item())
    # Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(pred - targets)).item()

    # Calculate R-squared
    ss_res = torch.sum((targets - pred) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = 1 - ss_res.item() / (ss_tot.item() + 1e-8)  # added epsilon to avoid division by zero

    return mse.item(), rmse, mae, r2


def train(model, epochs, batch_size, input_dim, device):
    model.train()  # Set model to training mode
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        # Generate synthetic data each iteration
        x, targets = generate_data(batch_size, input_dim)
        x, targets = x.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate additional metrics every 100 epochs (or on the final epoch)
        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            mse, rmse, mae, r2 = calculate_metrics(outputs, targets)
            print(f"Epoch {epoch + 1}/{epochs} -- MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")


def tune_train_func(config, checkpoint_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim =  100  # Dimension for the input logit vector
    hidden_dims = [640,1280]  # Hidden layer sizes
    epochs = 10000 # Reduced epochs for tuning

    model = SoftmaxApprox(input_dim, hidden_dims, dropout_prob=config["dropout_prob"])
    #model = torch.load("model.pth", map_location=device,weights_only=False)
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"]
    )
    loss_fn = torch.nn.MSELoss()

    best_r2 = -float('inf')  # Initialize best R²
    for epoch in range(epochs):
        # Generate synthetic training data using current batch size
        x, targets = generate_data(config["batch_size"], input_dim)
        x, targets = x.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        # Compute additional metrics every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            mse, rmse, mae, r2 = calculate_metrics(outputs, targets)
            metrics_dict = {
                "loss": loss.item(),
                "r2": r2,
                "mse": mse,
                "rmse": rmse,
                "mae": mae
            }
            # If current r2 is the best, save a checkpoint
            if r2 > best_r2:
                best_r2 = r2
                torch.save(model,"model.pth")
                torch.save(model.state_dict(), "softmax_weights.pth")

            print(f"Epoch {epoch + 1}/{epochs} -- Loss: {loss.item():.6f}, R²: {r2:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
            tune.report(metrics=metrics_dict)


# Define the grid search hyperparameter configuration.
search_space = {
    "lr": tune.grid_search([1e-3, 5e-4, 1e-4]),
    "batch_size": tune.grid_search([16,32, 64, 128]),
    "dropout_prob": tune.grid_search([0.1, 0.2, 0.3]),
    "beta1": tune.grid_search([0.9, 0.99]),
    "beta2": tune.grid_search([0.999, 0.9999]),
    "weight_decay": tune.grid_search([0.0, 1e-4, 1e-3])
}

trainable_with_resources = tune.with_resources(tune_train_func,{"cpu": 8, "gpu": 1})
stopper = stopper.TrialPlateauStopper(metric="r2", mode="max", grace_period=7,std=0.02,num_results=5)
asha_scheduler = ASHAScheduler(
    metric="r2",
    mode="max",
    grace_period=10,
    reduction_factor=5,
    time_attr="epoch",
    max_t=500
)

exp_dir = "./experiments"
exp_run = 0
for dir in os.listdir(exp_dir):
    name, exp = dir.split("_")
    exp_run += 1

exp_path = exp_dir + "/exp_" + str(exp_run) + "/"
os.makedirs(exp_path)

tuner = tune.Tuner(
    trainable_with_resources,
    param_space=search_space,

    tune_config=tune.TuneConfig(
        num_samples=1,
        scheduler=asha_scheduler,
        reuse_actors=True
    ),
    run_config=ray.train.RunConfig(
        stop=stopper,
        storage_path=Path(exp_path).resolve().as_posix(),
        storage_filesystem=pyarrow.fs.LocalFileSystem(use_mmap=False),
        checkpoint_config=ray.train.CheckpointConfig(

            num_to_keep=1,
            checkpoint_score_attribute="r2",
            checkpoint_score_order="max",
        ),
        failure_config=ray.train.FailureConfig(
            fail_fast=True,
        ),

    ),
)

# res = tuner.fit()
# best_res = res.get_best_result(metric="r2", mode="max")
# print(f"best : {best_res.metrics}")
# print(best_res.config)

tune_train_func({'lr': 0.001, 'batch_size': 64, 'dropout_prob': 0.1, 'beta1': 0.9, 'beta2': 0.9999, 'weight_decay': 0.0})



# Launch the grid search experiment with Ray Tune.
