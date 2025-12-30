import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
from pathlib import Path


class ANNVisualization:
    """Visualize ANN yield predictions and training metrics"""
    
    def plot_yield_prediction(self, actual_yields: list, predicted_yields: list, title: str = "ANN Yield Prediction: Actual vs Predicted"):
        """
        Plot actual vs predicted yield values with perfect prediction line
        
        Args:
            actual_yields: List of actual yield values
            predicted_yields: List of predicted yield values
            title: Plot title
        """
        actual = np.array(actual_yields)
        predicted = np.array(predicted_yields)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(actual, predicted, alpha=0.6, s=80, color='#2E86AB', edgecolors='black', linewidth=0.5, label='Predictions')
        
        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Metrics
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        ax.set_xlabel('Actual Yield Score', fontweight='bold', fontsize=12)
        ax.set_ylabel('Predicted Yield Score', fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add metrics text box
        metrics_text = f'R² Score: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}'
        ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self, train_loss: list, val_loss: list, 
                             train_accuracy: list = None, val_accuracy: list = None,
                             epochs: list = None):
        """
        Plot training vs validation loss and optionally accuracy
        
        Args:
            train_loss: Training loss values
            val_loss: Validation loss values
            train_accuracy: Training accuracy values (optional)
            val_accuracy: Validation accuracy values (optional)
            epochs: Epoch numbers (optional, auto-generated if None)
        """
        if epochs is None:
            epochs = list(range(1, len(train_loss) + 1))
        
        has_accuracy = train_accuracy is not None and val_accuracy is not None
        
        if has_accuracy:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle('ANN Training History', fontsize=16, fontweight='bold')
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            axes = [ax]
            fig.suptitle('ANN Training vs Validation Loss', fontsize=14, fontweight='bold')
        
        # Plot 1: Loss
        ax1 = axes[0]
        ax1.plot(epochs, train_loss, 'o-', linewidth=2, markersize=4, 
                color='#2E86AB', label='Training Loss', alpha=0.8)
        ax1.plot(epochs, val_loss, 's-', linewidth=2, markersize=4,
                color='#A23B72', label='Validation Loss', alpha=0.8)
        ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Loss', fontweight='bold', fontsize=11)
        ax1.set_title('Training vs Validation Loss', fontweight='bold', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add final loss values as annotation
        final_train_loss = train_loss[-1]
        final_val_loss = val_loss[-1]
        ax1.annotate(f'{final_train_loss:.4f}', 
                    xy=(epochs[-1], final_train_loss),
                    xytext=(-30, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        ax1.annotate(f'{final_val_loss:.4f}', 
                    xy=(epochs[-1], final_val_loss),
                    xytext=(-30, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Plot 2: Accuracy (if provided)
        if has_accuracy:
            ax2 = axes[1]
            ax2.plot(epochs, train_accuracy, 'o-', linewidth=2, markersize=4,
                    color='#06A77D', label='Training Accuracy', alpha=0.8)
            ax2.plot(epochs, val_accuracy, 's-', linewidth=2, markersize=4,
                    color='#D5573B', label='Validation Accuracy', alpha=0.8)
            ax2.set_xlabel('Epoch', fontweight='bold', fontsize=11)
            ax2.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
            ax2.set_title('Training vs Validation Accuracy', fontweight='bold', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Add final accuracy values
            final_train_acc = train_accuracy[-1]
            final_val_acc = val_accuracy[-1]
            ax2.annotate(f'{final_train_acc:.4f}', 
                        xy=(epochs[-1], final_train_acc),
                        xytext=(-30, -20), textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            ax2.annotate(f'{final_val_acc:.4f}', 
                        xy=(epochs[-1], final_val_acc),
                        xytext=(-30, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        return fig
    
    def plot_combined_ann_analysis(self, actual_yields: list, predicted_yields: list,
                                  train_loss: list, val_loss: list,
                                  train_accuracy: list = None, val_accuracy: list = None):
        """
        Plot all ANN metrics in one comprehensive figure
        
        Args:
            actual_yields: List of actual yield values
            predicted_yields: List of predicted yield values
            train_loss: Training loss values
            val_loss: Validation loss values
            train_accuracy: Training accuracy values (optional)
            val_accuracy: Validation accuracy values (optional)
        """
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('ANN Model Comprehensive Analysis', fontsize=18, fontweight='bold')
        
        # Define grid
        if train_accuracy and val_accuracy:
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        else:
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        actual = np.array(actual_yields)
        predicted = np.array(predicted_yields)
        
        # Plot 1: Actual vs Predicted
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(actual, predicted, alpha=0.6, s=80, color='#2E86AB', 
                   edgecolors='black', linewidth=0.5)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax1.set_xlabel('Actual Yield Score', fontweight='bold')
        ax1.set_ylabel('Predicted Yield Score', fontweight='bold')
        ax1.set_title('Actual vs Predicted', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training vs Validation Loss
        ax2 = fig.add_subplot(gs[0, 1])
        epochs = list(range(1, len(train_loss) + 1))
        ax2.plot(epochs, train_loss, 'o-', linewidth=2, markersize=4, 
                color='#2E86AB', label='Training Loss', alpha=0.8)
        ax2.plot(epochs, val_loss, 's-', linewidth=2, markersize=4,
                color='#A23B72', label='Validation Loss', alpha=0.8)
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Loss', fontweight='bold')
        ax2.set_title('Training vs Validation Loss', fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        ax3 = fig.add_subplot(gs[0, 2])
        residuals = actual - predicted
        ax3.scatter(predicted, residuals, alpha=0.6, s=80, color='#F18F01',
                   edgecolors='black', linewidth=0.5)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted Yield Score', fontweight='bold')
        ax3.set_ylabel('Residuals', fontweight='bold')
        ax3.set_title('Residual Plot', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Metrics Summary (Text)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        metrics_text = f"""
        Prediction Metrics:
        ─────────────────────
        R² Score:        {r2:.4f}
        RMSE:            {rmse:.4f}
        MAE:             {mae:.4f}
        MSE:             {mse:.4f}
        
        Loss Metrics:
        ─────────────────────
        Final Train Loss: {train_loss[-1]:.4f}
        Final Val Loss:   {val_loss[-1]:.4f}
        """
        
        ax4.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot 5: Loss Distribution Histogram
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(residuals, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Residual Value', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('Residual Distribution', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Accuracy (if provided)
        if train_accuracy and val_accuracy:
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.plot(epochs, train_accuracy, 'o-', linewidth=2, markersize=4,
                    color='#06A77D', label='Training Accuracy', alpha=0.8)
            ax6.plot(epochs, val_accuracy, 's-', linewidth=2, markersize=4,
                    color='#D5573B', label='Validation Accuracy', alpha=0.8)
            ax6.set_xlabel('Epoch', fontweight='bold')
            ax6.set_ylabel('Accuracy', fontweight='bold')
            ax6.set_title('Training vs Validation Accuracy', fontweight='bold')
            ax6.legend(fontsize=9)
            ax6.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def load_training_history(history_file: str):
        """
        Load training history from JSON file
        
        Args:
            history_file: Path to history JSON file
            
        Returns:
            Dictionary with training metrics
        """
        with open(history_file, 'r') as f:
            history = json.load(f)
        return history
    
    @staticmethod
    def save_training_history(history: dict, output_file: str):
        """
        Save training history to JSON file
        
        Args:
            history: Dictionary with training metrics
            output_file: Path to save JSON file
        """
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=2)


# Example usage
if __name__ == "__main__":
    viz = ANNVisualization()
    
    # Example 1: Plot yield predictions
    print("Generating yield prediction plot...")
    actual_yields = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                    0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    predicted_yields = [0.12, 0.18, 0.32, 0.38, 0.52, 0.62, 0.68, 0.82, 0.88, 0.98,
                       0.14, 0.24, 0.36, 0.42, 0.58, 0.64, 0.72, 0.86, 0.92]
    
    fig1 = viz.plot_yield_prediction(actual_yields, predicted_yields)
    plt.show()
    
    # Example 2: Plot training history
    print("Generating training history plot...")
    train_loss = [0.085, 0.065, 0.045, 0.032, 0.024, 0.018, 0.014, 0.011, 0.009, 0.008,
                 0.007, 0.006, 0.006, 0.005, 0.005, 0.004]
    val_loss = [0.080, 0.062, 0.042, 0.032, 0.026, 0.022, 0.020, 0.018, 0.017, 0.017,
                0.017, 0.017, 0.016, 0.016, 0.016, 0.016]
    train_acc = [0.5, 0.65, 0.75, 0.82, 0.87, 0.90, 0.92, 0.93, 0.94, 0.95,
                0.95, 0.96, 0.96, 0.96, 0.97, 0.97]
    val_acc = [0.48, 0.62, 0.72, 0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.92,
              0.92, 0.92, 0.93, 0.93, 0.93, 0.93]
    
    fig2 = viz.plot_training_history(train_loss, val_loss, train_acc, val_acc)
    plt.show()
    
    # Example 3: Combined analysis
    print("Generating combined analysis plot...")
    fig3 = viz.plot_combined_ann_analysis(actual_yields, predicted_yields,
                                         train_loss, val_loss, train_acc, val_acc)
    plt.show()
