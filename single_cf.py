from sklearn.metrics import confusion_matrix
import torch
import itertools
# import torch.nn as nn
# import torch.optim as optim
from data import create_data_loaders
import models_vit
# from util.pos_embed import interpolate_pos_embed
# from timm.models.layers import trunc_normal_
import numpy as np
import os
import logging
from torchvision import models
import matplotlib.pyplot as plt

def calculate_accuracy(model, data_loader, device='cuda',task='healthy',threshold =0.5):
    
    model = model.to(device)
    model.eval()
    # print(task)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels[task].unsqueeze(1).float().to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            predictions = (outputs > threshold).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

def plot_confusion_matrix(cm, classes, ax, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    """
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

def plot_confusion_matrices(confusion_matrices, tasks, filename=None):
    """
    This function plots multiple confusion matrices in the same figure.
    """
    num_tasks = len(tasks)
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))  # Adjust the number of rows and columns as needed

    for i, (task, cm) in enumerate(confusion_matrices.items()):
        row = i // 4
        col = i % 4
        plot_confusion_matrix(cm, classes=["Negative", "Positive"], ax=axs[row, col], title=f'Confusion matrix for {task}')

    plt.tight_layout()
    
    # Save the figure if filename is provided
    if filename:
        plt.savefig(filename)
    
    plt.show()

def get_confusion_matrix(model, test_loader, device='cuda',task="healthy",threshold=0.5):
    model.eval()
    
    with torch.no_grad():
        # for task in tasks:
        all_labels = []
        all_predictions = []

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels[task].unsqueeze(1).float().to(device)
            all_labels.extend(labels.cpu().numpy())
            outputs = model(inputs)
            predictions = (outputs > threshold).float()
            all_predictions.extend(predictions.cpu().numpy())

        cf_matrix = confusion_matrix(all_labels, all_predictions)

    return cf_matrix


# Main function
def main():
    # Define hyperparameters
    batch_size = 32
    # logging.basicConfig(filename='training_logs_multiclass_models_retfound_confusion_matrix.txt', level=logging.INFO,
                    # format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Define paths and filenames
    csv_file = '/home/tejadhith/Project/OCT/Dataset/28-09-2023_download.csv'
    root_dir = '/home/tejadhith/Project/OCT/Dataset/segregated_28-sep-2023_kath'
    model_save_root_dir = '/home/tejadhith/Project/OCT/models_retfound'

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(csv_file, root_dir, batch_size=batch_size)
    print("Length of Train Dataset : ",len(train_loader.dataset))
    print("Length of Val Dataset : ",len(val_loader.dataset))
    print("Length of Test Dataset : ",len(test_loader.dataset))

    # call the model
    model = models_vit.__dict__['vit_large_patch16'](
        num_classes=1,
        drop_path_rate=0,
        global_pool=True,
    )

    # PATH = "/home/tejadhith/Project/OCT/models_retfound"
    # model.load_state_dict(torch.load(PATH,map_location="cpu"))
    # Define loss function and optimizer
    # model = OCT_Classifier(num_classes=8)

    tasks = ["foveal_scan","healthy","srf","irf","drusen","ped","hdots","hfoci"]
    confusion_matrices = {}
    device = "cuda"
    for i, task in enumerate(tasks):
        model_save_dir = os.path.join(model_save_root_dir,task)
        model_file = os.path.join(model_save_dir,"best_model.pth")
        model.load_state_dict(torch.load(model_file,map_location="cpu"))
        model = model.to(device)
        test_accuracy_final = calculate_accuracy(model, test_loader, device='cuda',task=task)

        print(f" Final Test Accuracy ({task}): {test_accuracy_final:.4f}")

        confusion_matrices[task] = get_confusion_matrix(model, test_loader, 'cuda', task, 0.5)
    
    
    # Save confusion matrices as images
    save_path = 'confusion_matrices_single_0_50.png'

    plot_confusion_matrices(confusion_matrices, tasks,save_path)


if __name__ == "__main__":
    main()
# Call this function after your main training loop to get confusion matrices
