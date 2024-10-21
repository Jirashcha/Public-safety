import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay
from ultralytics import YOLO
import torch
from datetime import datetime
import numpy as np
import multiprocessing

# Check if a GPU is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Define path for data.yaml
data_path = 'data.yaml'
epochs = 10               # Set a reasonable number of epochs
workers = 8               # Number of device workers (0 - 16, commonly 4 - 16)
batch_size = 16           # Increase batch size if GPU memory allows
img_size = 640            # Image size
save_period = epochs // 2

# Create YOLOv8 model and move it to the appropriate device
model = YOLO('yolo11n.pt')
model.to(device)

def main():
    res = model.train(data=data_path, epochs=epochs, imgsz=img_size)

if __name__ == '__main__':
    # multiprocessing.freeze_support()    # required for Windows if ur freezing the script
    main()

exit()

# Function to save metrics
def save_metrics(metrics, timestamp, filename='metrics.txt'):
    output_filename = f"{filename.split('.')[0]}_{timestamp}.txt"
    with open(output_filename, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, timestamp, filename='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    output_filename = f"confusion_matrix_{timestamp}.png"
    plt.savefig(output_filename)

# Get current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metrics = {}
y_true, y_pred = [], []

results = model.train(data=data_path, epochs=epochs, imgsz=img_size, batch=batch_size, 
                        workers=workers, save_period=save_period, device=device, name=f"trained_model_{timestamp}")

try:
    # Train the model


    # Assuming val_labels and val_preds are obtained through some post-processing
    y_true, y_pred = [], []  # Placeholder: replace with actual method to extract labels and predictions if available

    if y_true and y_pred:  # Check if labels and predictions were obtained successfully
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics = {
            'Accuracy': accuracy,
            'F1 Score': f1
        }
        save_metrics(metrics, timestamp)
        plot_confusion_matrix(y_true, y_pred, timestamp)
    else:
        print("y_true and y_pred are unavailable; skipping metrics and confusion matrix plotting.")

    # Save loss graph
    if 'train_loss' in results and 'val_loss' in results:
        plt.figure()
        plt.plot(results['train_loss'], label='Train Loss')
        plt.plot(results['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(f'loss_curve_{timestamp}.png')
    else:
        print("Loss data is unavailable; skipping loss plot.")

    # Save the trained model with timestamp
    model.save(f'trained_model_{timestamp}.pt')

except Exception as e:
    print(f"An error occurred: {e}")
    # Save the model and metrics if an error occurs
    model.save(f'interrupted_model_{timestamp}.pt')
    if metrics:  # Only save if metrics were calculated
        save_metrics(metrics, timestamp)
    if y_true and y_pred:  # Only plot if we have valid labels and predictions
        plot_confusion_matrix(y_true, y_pred, timestamp)
