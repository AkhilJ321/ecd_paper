import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
import time
from datetime import datetime
import os

# Custom Dataset Class
class EngineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Base ANN Model
class BaseANNModel(nn.Module):
    def __init__(self, activation_fn=nn.ReLU()):
        super(BaseANNModel, self).__init__()
        self.input_dim = 7
        self.hidden_dim = 30
        self.output_dim = 3
        
        # Layers
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        
        self.layer2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        
        self.layer3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)
        
        self.layer4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn4 = nn.BatchNorm1d(self.hidden_dim)
        
        self.layer5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn5 = nn.BatchNorm1d(self.hidden_dim)
        
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.activation = activation_fn
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.dropout(self.activation(self.bn1(self.layer1(x))))
        x = self.dropout(self.activation(self.bn2(self.layer2(x))))
        x = self.dropout(self.activation(self.bn3(self.layer3(x))))
        x = self.dropout(self.activation(self.bn4(self.layer4(x))))
        x = self.dropout(self.activation(self.bn5(self.layer5(x))))
        x = self.output_layer(x)
        return x

# Performance Metrics Class
class PerformanceMetrics:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        
    def update(self, predictions, targets):
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
            
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        
    def compute_metrics(self):
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))
        metrics['exhaust_rmse'] = rmse[0]
        metrics['cw1_rmse'] = rmse[1]
        metrics['cw2_rmse'] = rmse[2]
        metrics['mean_rmse'] = np.mean(rmse)
        
        # R²
        r2_scores = []
        for i in range(targets.shape[1]):
            r2 = r2_score(targets[:, i], predictions[:, i])
            r2_scores.append(r2)
        
        metrics['exhaust_r2'] = r2_scores[0]
        metrics['cw1_r2'] = r2_scores[1]
        metrics['cw2_r2'] = r2_scores[2]
        metrics['mean_r2'] = np.mean(r2_scores)
        
        # MAE
        mae = np.mean(np.abs(predictions - targets), axis=0)
        metrics['exhaust_mae'] = mae[0]
        metrics['cw1_mae'] = mae[1]
        metrics['cw2_mae'] = mae[2]
        metrics['mean_mae'] = np.mean(mae)
        
        return metrics

# Training and Evaluation Functions
class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 criterion, optimizer, device, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.metrics = PerformanceMetrics()
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_metrics': [], 'val_metrics': []
        }
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        self.metrics.reset()
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.metrics.update(outputs, targets)
            
        epoch_loss = total_loss / len(self.train_loader)
        epoch_metrics = self.metrics.compute_metrics()
        
        return epoch_loss, epoch_metrics
    
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        self.metrics.reset()
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                self.metrics.update(outputs, targets)
        
        avg_loss = total_loss / len(loader)
        metrics = self.metrics.compute_metrics()
        
        return avg_loss, metrics
    
    def train(self, epochs, early_stopping_patience=50):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_metrics = self.train_epoch()
            
            # Validation
            val_loss, val_metrics = self.evaluate(self.val_loader)
            
            # Update learning rate scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Print progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"Train RMSE: {train_metrics['mean_rmse']:.4f}, "
                      f"Val RMSE: {val_metrics['mean_rmse']:.4f}")
                print(f"Train R²: {train_metrics['mean_r2']:.4f}, "
                      f"Val R²: {val_metrics['mean_r2']:.4f}")
        
        return self.history

# Visualization Functions
class VisualizationTools:
    @staticmethod
    def plot_training_history(history, save_path=None):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss History')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # RMSE plot
        train_rmse = [m['mean_rmse'] for m in history['train_metrics']]
        val_rmse = [m['mean_rmse'] for m in history['val_metrics']]
        ax2.plot(train_rmse, label='Training RMSE')
        ax2.plot(val_rmse, label='Validation RMSE')
        ax2.set_title('RMSE History')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE')
        ax2.legend()
        
        # R² plot
        train_r2 = [m['mean_r2'] for m in history['train_metrics']]
        val_r2 = [m['mean_r2'] for m in history['val_metrics']]
        ax3.plot(train_r2, label='Training R²')
        ax3.plot(val_r2, label='Validation R²')
        ax3.set_title('R² History')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('R²')
        ax3.legend()
        
        # MAE plot
        train_mae = [m['mean_mae'] for m in history['train_metrics']]
        val_mae = [m['mean_mae'] for m in history['val_metrics']]
        ax4.plot(train_mae, label='Training MAE')
        ax4.plot(val_mae, label='Validation MAE')
        ax4.set_title('MAE History')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MAE')
        ax4.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_prediction_comparison(y_true, y_pred, labels, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
            ax.plot([y_true[:, i].min(), y_true[:, i].max()], 
                   [y_true[:, i].min(), y_true[:, i].max()], 
                   'r--', lw=2)
            ax.set_title(f'{label} Predictions vs Actual')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

# Model Comparison Framework
class ModelComparison:
    def __init__(self, data_loaders, device):
        self.data_loaders = data_loaders
        self.device = device
        self.activation_functions = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        self.results = {}
        
    def train_and_evaluate(self, epochs=6000):
        for name, activation in self.activation_functions.items():
            print(f"\nTraining model with {name} activation function...")
            
            # Initialize model and training components
            model = BaseANNModel(activation).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
            
            # Train model
            trainer = ModelTrainer(
                model, self.data_loaders['train'], 
                self.data_loaders['val'], self.data_loaders['test'],
                criterion, optimizer, self.device, scheduler
            )
            
            history = trainer.train(epochs)
            
            # Final evaluation
            test_loss, test_metrics = trainer.evaluate(self.data_loaders['test'])
            
            # Store results
            self.results[name] = {
                'model': model,
                'history': history,
                'test_metrics': test_metrics
            }
            
    def compare_results(self):
        comparison_df = pd.DataFrame()
        
        for name, result in self.results.items():
            metrics = result['test_metrics']
            comparison_df[name] = [
                metrics['mean_rmse'],
                metrics['mean_r2'],
                metrics['mean_mae']
            ]
            
        comparison_df.index = ['RMSE', 'R²', 'MAE']
        return comparison_df
    
    def plot_comparison(self, save_path=None):
        comparison_df = self.compare_results()
        
        plt.figure(figsize=(12, 6))
        comparison_df.plot(kind='bar')
        plt.title('Model Comparison Across Activation Functions')
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.legend(title='Activation Functions')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

# Main execution function
def main(data_path, results_path='results'):
    # Create results directory
    os.makedirs(results_path, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    df = pd.read_csv(data_path)
    
    # Prepare features and targets
    X = df.iloc[:, 1:8].values  # Features
    y = df.iloc[:, -3:].values  # Targets
    
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Split the data (70% train, 15% validation, 15% test)
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_scaled, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)
    
    # Create datasets
    train_dataset = EngineDataset(X_train, y_train)
    val_dataset = EngineDataset(X_val, y_val)
    test_dataset = EngineDataset(X_test, y_test)
    
    # Create dataloaders
    data_loaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32),
        'test': DataLoader(test_dataset, batch_size=32)
    }
    
    # Initialize model comparison
    model_comparison = ModelComparison(data_loaders, device)
    
    # Train and evaluate all models
    print("Starting model training and comparison...")
    model_comparison.train_and_evaluate(epochs=6000)
    
    # Save and display results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Compare results
    comparison_df = model_comparison.compare_results()
    comparison_df.to_csv(f"{results_path}/model_comparison_{timestamp}.csv")
    print("\nModel Comparison Results:")
    print(comparison_df)
    
    # Plot comparison
    model_comparison.plot_comparison(f"{results_path}/comparison_plot_{timestamp}.png")
    
    # For each activation function, create detailed plots
    for name, result in model_comparison.results.items():
        print(f"\nDetailed results for {name} activation function:")
        
        # Create plots directory for this model
        model_path = f"{results_path}/{name}_{timestamp}"
        os.makedirs(model_path, exist_ok=True)
        
        # Plot training history
        VisualizationTools.plot_training_history(
            result['history'],
            save_path=f"{model_path}/training_history.png"
        )
        
        # Get predictions for test set
        model = result['model']
        model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loaders['test']:
                X_batch = X_batch.to(device)
                pred = model(X_batch).cpu().numpy()
                y_pred.extend(pred)
                y_true.extend(y_batch.numpy())
                
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        # Inverse transform predictions and true values
        y_pred_original = scaler_y.inverse_transform(y_pred)
        y_true_original = scaler_y.inverse_transform(y_true)
        
        # Plot prediction comparison
        VisualizationTools.plot_prediction_comparison(
            y_true_original, y_pred_original,
            ['Exhaust Temperature', 'CW1 Temperature', 'CW2 Temperature'],
            save_path=f"{model_path}/prediction_comparison.png"
        )
        
        # Save detailed metrics
        metrics = result['test_metrics']
        with open(f"{model_path}/detailed_metrics.txt", 'w') as f:
            f.write(f"Results for {name} activation function:\n\n")
            f.write("Temperature\tRMSE\tR²\tMAE\n")
            f.write(f"Exhaust\t{metrics['exhaust_rmse']:.4f}\t{metrics['exhaust_r2']:.4f}\t{metrics['exhaust_mae']:.4f}\n")
            f.write(f"CW1\t{metrics['cw1_rmse']:.4f}\t{metrics['cw1_r2']:.4f}\t{metrics['cw1_mae']:.4f}\n")
            f.write(f"CW2\t{metrics['cw2_rmse']:.4f}\t{metrics['cw2_r2']:.4f}\t{metrics['cw2_mae']:.4f}\n")
            f.write(f"\nMean Metrics:\n")
            f.write(f"Mean RMSE: {metrics['mean_rmse']:.4f}\n")
            f.write(f"Mean R²: {metrics['mean_r2']:.4f}\n")
            f.write(f"Mean MAE: {metrics['mean_mae']:.4f}\n")
        
        # Save model
        torch.save(model.state_dict(), f"{model_path}/model.pth")
    
    print(f"\nAll results have been saved to {results_path}")
    return model_comparison

if __name__ == "__main__":
    # Example usage
    data_path = "data.csv"  # Replace with your data path
    model_comparison = main(data_path)