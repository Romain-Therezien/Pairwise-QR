# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from matplotlib.patches import Patch
import time
# ---- Settings ----
np.random.seed(42)
def pinball_loss(y_true, y_pred, quantile):
    u = y_true - y_pred
    loss = torch.mean(torch.max(quantile * u, (quantile - 1) * u))
    return loss

def generate_data(n_samples):
    """
    Generate pairwise data with customizable score function.

    Args:
        n_samples (int): Number of samples.
    Returns:
        Z_pairs (torch.Tensor): Tensor of shape (num_pairs, 2) of Z pairs.
        pair_scores (torch.Tensor): Tensor of shape (num_pairs,) of scores.
    """
    Z = np.random.uniform(-1.5, 1.5, n_samples) #np.random.uniform(-1, 1, n_samples)
    mu = 0.1*Z #np.sin(0.5*np.pi * Z)
    sigma = 0.3*np.abs(Z) #np.abs(Z)**2 +0.3 #0.1 + 0.5 * np.abs(Z)
    X = Z + mu + sigma * np.random.normal(0, 1, len(Z))
    X = torch.tensor(X, dtype=torch.float32)
    score = torch.sin((X[:, None] + X[None, :]))
    #np.sin((X[:, None] * X[None, :])) #np.sin((X[:, None] - X[None, :])/2)
    triu_indices = torch.triu_indices(n_samples, n_samples, offset=1)
    pair_scores = score[triu_indices[0], triu_indices[1]]
    pair_Z = np.stack([Z[triu_indices[0]], Z[triu_indices[1]]], axis=1)
    return torch.Tensor(pair_Z), pair_scores


# -----------------------------
# Shared noise (PAIRWISE FIX)
# -----------------------------
def build_shared_noise(n_pairs, n_mc=1000, seed=42):
    rng = np.random.default_rng(seed)
    eps_i = rng.normal(size=(n_pairs, n_mc)).astype(np.float32)
    eps_j = rng.normal(size=(n_pairs, n_mc)).astype(np.float32)
    return eps_i, eps_j


# -----------------------------
# True quantile computation
# -----------------------------
def compute_true_quantile_chunked(Z, quantiles, n_mc=5000, chunk_size=250):
    rng = np.random.default_rng(42)

    quantiles = np.atleast_1d(quantiles)
    n_pairs = Z.shape[0]

    samples = np.empty((n_pairs, n_mc), dtype=np.float32)

    Z_i = Z[:, 0][:, None]
    Z_j = Z[:, 1][:, None]

    mu_i = 0.1 * Z_i
    mu_j = 0.1 * Z_j

    sigma_i = 0.3 * np.abs(Z_i)
    sigma_j = 0.3 * np.abs(Z_j)

    for start in range(0, n_mc, chunk_size):
        end = min(start + chunk_size, n_mc)
        m = end - start

        eps_i = rng.normal(size=(n_pairs, m)).astype(np.float32)
        eps_j = rng.normal(size=(n_pairs, m)).astype(np.float32)

        X_i = Z_i + mu_i + sigma_i * eps_i
        X_j = Z_j + mu_j + sigma_j * eps_j

        samples[:, start:end] = np.sin(X_i + X_j)

    return np.stack([
        np.quantile(samples, q, axis=1)
        for q in quantiles
    ])

def training_tree(X_train, Y_train, X_val, Y_val, quantile=0.5):
    model =  LGBMRegressor(
        objective='quantile',
        alpha=quantile,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        n_estimators=70,
        verbosity=-1,
    )
    model.fit(
        X_train, Y_train,
        eval_set=[(X_train, Y_train), (X_val, Y_val)],
        eval_metric='quantile',
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=0)
        ],
    )
    return model

class NNQuantileRegressor(nn.Module):
    def __init__(self, input_dim, hidden_units, quantile):
        super(NNQuantileRegressor, self).__init__()
        self.quantile = quantile
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze()
    

def train_NN(quantile, X_train, Y_train, X_test=None, Y_test=None, n_epochs=1000, log_loss=True):
    model = NNQuantileRegressor(input_dim=X_train.shape[1], hidden_units=32, quantile=quantile)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    loss_train, loss_test = [], []
    for epoch in range(n_epochs):
        model.train()
        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = pinball_loss(Y_batch, outputs, quantile)
            loss.backward()
            optimizer.step()
        if X_test is not None and Y_test is not None:
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
                test_outputs = model(X_test_tensor)
                test_loss = pinball_loss(Y_test_tensor, test_outputs, quantile)
                loss_test.append(test_loss.item())
        if log_loss == True:
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train_tensor)
                train_loss = pinball_loss(Y_train_tensor, train_outputs, quantile)
                loss_train.append(train_loss.item())

    if X_test is None and Y_test is None:
        return model, loss_train
    else:
        return model, loss_train, loss_test

def model_train(X_train, Y_train, X_test, Y_test, quantiles):
    # Train models for each quantile
    models = {}
    for quantile in quantiles:
        print(f"Training quantile {quantile}")
        nn_model, _, _ = train_NN(quantile, X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy(), n_epochs=n_epochs)
        models[quantile] = nn_model

    error_train = {key: [] for key in models.keys()}
    error_test = {key: [] for key in models.keys()}

    for quantile in quantiles:
        models_nn = models[quantile]
        # Predictions on train set
        preds_nn_train = models_nn(torch.tensor(X_train, dtype=torch.float32)).detach().numpy()
        error_train[quantile] = pinball_loss(Y_train, torch.tensor(preds_nn_train, dtype=torch.float32), quantile).item()
        # Predictions on test set
        preds_nn_test = models_nn(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
        error_test[quantile] = pinball_loss(Y_test, torch.tensor(preds_nn_test, dtype=torch.float32), quantile).item()
    return error_train, error_test

def model_uncertainty(quantiles, n_repeat=10):
    error_train_all = {key: [] for key in quantiles}
    error_test_all = {key: [] for key in quantiles}

    for repeat in range(n_repeat):
        print(f"Repeat {repeat+1}/{n_repeat}")
        X_train, Y_train = generate_data(n_train)
        X_test, Y_test = generate_data(n_test)
        error_train, error_test = model_train(X_train, Y_train, X_test, Y_test, quantiles)
        for quantile in quantiles:
            error_train_all[quantile].append(error_train[quantile])
            error_test_all[quantile].append(error_test[quantile])

    # Compute mean and std
    error_train_mean = {key: np.mean(error_train_all[key]) for key in quantiles}
    error_test_mean = {key: np.mean(error_test_all[key]) for key in quantiles}
    error_train_std = {key: np.std(error_train_all[key]) for key in quantiles}
    error_test_std = {key: np.std(error_test_all[key]) for key in quantiles}

    return error_train_mean, error_test_mean, error_train_std, error_test_std

# %% Generate data
n_train, n_test = 500, 100
noise_std = 0.1
n_epochs = 100
quantiles = [round(x, 2) for x in np.linspace(0, 1, 21)[1:-1]]  # exclude 0 and 1
nb_bootstraps = 10

X_train, Y_train = generate_data(n_train)
X_test, Y_test = generate_data(n_test)
# %% Train model for each quantile and evaluate with confidence intervals
error_train_mean, error_test_mean, error_train_std, error_test_std = model_uncertainty(quantiles)

# Plot errors
plt.figure(figsize=(12, 6))
plt.plot(quantiles, [error_train_mean[q] for q in quantiles], label='Train Error', marker='o')
plt.fill_between(quantiles, 
                 [error_train_mean[q] - error_train_std[q] for q in quantiles],
                 [error_train_mean[q] + error_train_std[q] for q in quantiles],
                 color='blue', alpha=0.2)
plt.plot(quantiles, [error_test_mean[q] for q in quantiles], label='Test Error', marker='o')
plt.fill_between(quantiles, 
                 [error_test_mean[q] - error_test_std[q] for q in quantiles],
                 [error_test_mean[q] + error_test_std[q] for q in quantiles],
                 color='orange', alpha=0.2)
plt.xlabel('Quantiles')
plt.ylabel('Pinball Loss')
plt.title('Pinball Loss Across Quantiles')
plt.legend()
plt.savefig('Pinball_loss_across_quantiles_nn.png')
plt.show()

# %% 3D plots
quantiles_to_plot = [0.1, 0.5, 0.9]
models = {}
for quantile in quantiles_to_plot:
    print(f"Training quantile {quantile} for 3D plot")
    nn_model, _, _ = train_NN(quantile, X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy(), n_epochs=n_epochs)
    models[quantile] = nn_model
models_to_plot = {q: models[q] for q in quantiles_to_plot}
# Prepare meshgrid for plotting
z_line = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 50)
z_prime_line = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 50)
Zg, Zpg = np.meshgrid(z_line, z_prime_line)

X_test, Y_test = generate_data(n_test)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter points
ax.scatter(X_test[:, 0], X_test[:, 1], Y_test, alpha=0.4, s=10, color='yellow', edgecolor='k')
# Plot surfaces for each quantile
colors = ['blue', 'orange', 'green',]
alphas = [0.3, 0.4, 0.5]
for (tau, model), c, a in zip(models_to_plot.items(), colors, alphas):
    # Flatten the grid to pass through model.predict
    grid_points = np.c_[Zg.ravel(), Zpg.ravel()]
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Sg = model(grid_points_tensor).numpy().reshape(Zg.shape)
    ax.plot_surface(Zg, Zpg, Sg, alpha=a, color=c)
# Axis labels and title
ax.set_xlabel('Z')
ax.set_ylabel("Z'")
ax.set_zlabel('S')
ax.view_init(elev=30, azim=-50)
plt.title(f'Neural Network Quantile Regression Surfaces')
# Custom legend
legend_elements = [Patch(facecolor='yellow', alpha=0.4, edgecolor='k', label='Scores')] + \
                [Patch(facecolor=c, alpha=a, label=f'Quantile {tau}') for tau, c, a in zip(quantiles_to_plot, colors, alphas)]
ax.legend(handles=legend_elements)
plt.savefig(f'Neural_Network_Quantile_Regression_Surfaces.pdf', dpi=300)
plt.show()
# %% Compare different models
results_pinball, results_abs_error = {}, {}
quantiles = np.linspace(0.05, 0.95, 19)
nb_bootstraps = 10
n_train = 1000
n_val = 200

np.random.seed(42)
X_val, Y_val = generate_data(n_val)
X_val_np = X_val

n_pairs_val = X_val_np.shape[0]

eps_i, eps_j = build_shared_noise(n_pairs_val, n_mc=5000, seed=42)

true_q_val = compute_true_quantile_chunked(
    X_val_np,
    quantiles,
    n_mc=5000,
    chunk_size=250
)

# %%
import time
seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
for bst in range(nb_bootstraps):
    seed = seeds[bst]
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_train, Y_train = generate_data(n_train)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    for q_idx, quantile in enumerate(quantiles):
        # ---- Define models per quantile ----
        models = {
            #'RandomForest': RandomForestRegressor(n_estimators=100),
            'GradientBoosting': GradientBoostingRegressor(loss='quantile', alpha=quantile, learning_rate=0.01, max_depth=5, min_samples_leaf=5, n_estimators=600, subsample=0.7),
            'LightGBM': LGBMRegressor(objective='quantile', alpha=quantile, learning_rate=0.05, max_depth=6, min_child_samples=20, num_leaves=63, reg_lambda=0, subsample=0.7),
            'NeuralNet': NNQuantileRegressor(input_dim=2, hidden_units=32, quantile=quantile),
        }

        for model_name, model in models.items():
            time_start = time.time()
            if model_name == 'NeuralNet':
                model, _= train_NN(quantile, X_train.numpy(), Y_train.numpy(), n_epochs=n_epochs)
                model.eval()
                with torch.no_grad():
                    preds = model(X_val_tensor).numpy()

            elif model_name == 'LightGBM':
                model.fit(
                    np.array(X_train), np.array(Y_train),
                    verbose=False,
                    callbacks=[
                        log_evaluation(period=0) 
                    ],
                )
                preds = model.predict(np.array(X_val))
            else:
                # GradientBoosting
                model.fit(np.array(X_train), np.array(Y_train))
                preds = model.predict(np.array(X_val))

            # Compute true quantile for evaluation
            err = np.mean(np.abs(true_q_val[q_idx] - preds))
            results_abs_error[(model_name, quantile, bst)] = err
            print(f"{bst+1} bootstrap, {model_name} (q={quantile}) Absolute Error: {err:.4f} - Time: {time.time() - time_start:.2f}s")
            error_pinball = pinball_loss(Y_val, torch.tensor(preds, dtype=torch.float32), quantile)
            results_pinball[(model_name, quantile, bst)] = error_pinball.item()
            print(f"{bst+1} bootstrap, {model_name} (q={quantile}) Pinball Loss: {error_pinball.item()}")
    print(f"Completed bootstrap {bst+1}/{nb_bootstraps}")
# %%
from collections import defaultdict
errors_by_model_quantile = defaultdict(list)
for (model, quantile, bst), error in results_abs_error.items():
    errors_by_model_quantile[(model, np.round(quantile, 2))].append(float(error))

# Step 2: Compute mean and CI
summary_by_model = defaultdict(lambda: {'quantiles': [], 'mean': [], 'ci_lower': [], 'ci_upper': []})

for (model, quantile), errors in errors_by_model_quantile.items():
    errors = np.array(errors)
    mean_error = np.mean(errors)
    ci_lower = np.percentile(errors, 2.5)
    ci_upper = np.percentile(errors, 97.5)
    summary_by_model[model]['quantiles'].append(quantile)
    summary_by_model[model]['mean'].append(mean_error)
    summary_by_model[model]['ci_lower'].append(ci_lower)
    summary_by_model[model]['ci_upper'].append(ci_upper)

# Step 3: Plot
plt.figure(figsize=(8,5))
for model, data in summary_by_model.items():
    # sort by quantile
    sorted_idx = np.argsort(data['quantiles'])
    quantiles_sorted = np.array(data['quantiles'])[sorted_idx]#[:-1]
    mean_sorted = np.array(data['mean'])[sorted_idx]#[:-1]
    ci_lower_sorted = np.array(data['ci_lower'])[sorted_idx]#[:-1]
    ci_upper_sorted = np.array(data['ci_upper'])[sorted_idx]#[:-1]
    plt.plot(quantiles_sorted, mean_sorted, marker='o', label=model)
    plt.fill_between(quantiles_sorted, ci_lower_sorted, ci_upper_sorted, alpha=0.2)

plt.xlabel('Quantile')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error by Model and Quantile with 95% CI')
plt.legend()
plt.grid(True)
plt.savefig('Model_Comparison_Absolute_Error.png')
plt.show()


# %% Compare Speed and accuracy for incomplete U-statistic vs complete U-statistic
def compare_speed_accuracy(n_train=1000, n_test=100, quantile=0.5, n_epochs=1000):
    X_train, Y_train = generate_data(n_train)
    X_test, Y_test = generate_data(n_test)
    # Complete U-statistic
    start_time = time.time()
    model_complete, loss_train_complete = train_NN(quantile, X_train.numpy(), Y_train.numpy(), n_epochs=n_epochs, log_loss=False)
    run_time_complete = time.time() - start_time
    # Incomplete U-statistic: sample pairs
    n_pairs = int(n_train * np.log(n_train))
    indices = np.random.choice(len(X_train), n_pairs)
    X_train_incomplete = torch.stack([X_train[indices, 0], X_train[indices, 1]], dim=1)
    Y_train_incomplete = Y_train[indices]
    start_time = time.time()
    model_incomplete, loss_train_incomplete = train_NN(quantile, X_train_incomplete.numpy(), Y_train_incomplete.numpy(), n_epochs=n_epochs, log_loss=False)
    run_time_incomplete = time.time() - start_time
    # Evaluate accuracy
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    # Complete U-statistic predictions
    model_complete.eval()
    with torch.no_grad():
        preds_complete = model_complete(X_test_tensor).numpy()
    error_complete = pinball_loss(Y_test_tensor, torch.tensor(preds_complete, dtype=torch.float32), quantile)
    # Incomplete U-statistic predictions
    model_incomplete.eval()
    with torch.no_grad():  
        preds_incomplete = model_incomplete(X_test_tensor).numpy()
    error_incomplete = pinball_loss(Y_test_tensor, torch.tensor(preds_incomplete, dtype=torch.float32), quantile)
    df = pd.DataFrame({
        'Method': ['Complete U-statistic', 'Incomplete U-statistic'],
        'Training Time (s)': [run_time_complete, run_time_incomplete],
        'Pinball Loss': [error_complete.item(), error_incomplete.item()]
    })
    return df
results_df = compare_speed_accuracy(n_train=1000, n_test=100, quantile=0.5, n_epochs=100)
print(results_df)
# %%
