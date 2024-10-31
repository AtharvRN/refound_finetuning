import torch
import torch.nn as nn
import torch.optim as optim
from data import create_data_loaders,create_k_fold_data_loaders
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import os
import logging
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
# Configure logger to write to a file

def calculate_accuracy(model, data_loader, device='cuda', tasks= ["foveal_scan","healthy","srf","irf","drusen","hdots","hfoci","ped"], threshold=0.5):
    model = model.to(device)
    model.eval()
    
    num_tasks = len(tasks)
    correct = [0] * num_tasks
    total = [0] * num_tasks
    accuracies = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}  # Move all labels to device
            labels = torch.stack([labels[key] for key in labels], dim=1)
    
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
        # correct = 0
        # total = 0
            for i in range(num_tasks):
                predictions = (outputs[:,i] > threshold).float()
                correct[i] += (predictions == labels[:,i]).sum().item()
                total[i] += labels.size(0)

    accuracies = [correct[i] / total[i] for i in range(num_tasks)]
    return accuracies

def custom_loss(output, target):
    # Assuming output and target are both of shape (batch_size, 8)
    loss = 0.0
    criterion = nn.BCELoss()  # Sum the losses across the batch
    for i in range(output.size(1)):  # Iterate over each dimension of the output
        loss += criterion(output[:, i], target[:, i].float()) 
    return loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', tasks=["foveal_scan", "healthy", "srf", "irf", "drusen", "hdots", "hfoci", "ped"], save_freq=5, model_save_dir=""):
    model.to(device)
    train_losses = []
    val_losses = []
    train_accuracies = [[] for _ in range(len(tasks))]
    val_accuracies = [[] for _ in range(len(tasks))]
    best_val_loss = 1e10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}  # Move all labels to device
            labels = torch.stack([labels[key] for key in labels], dim=1)
        
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            loss = custom_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

        # Calculate training accuracy
        train_accuracies_epoch = calculate_accuracy(model, train_loader, device, tasks)
        for i, task in enumerate(tasks):
            train_accuracies[i].append(train_accuracies_epoch[i])
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Accuracy ({task}): {train_accuracies_epoch[i]:.4f}")

        # Validate the model
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = {key: value.to(device) for key, value in labels.items()}  # Move all labels to device
                labels = torch.stack([labels[key] for key in labels], dim=1)
            
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                loss = custom_loss(outputs, labels)
                val_loss += loss.item()
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {epoch_val_loss:.4f}")

        # Calculate validation accuracy
        val_accuracies_epoch = calculate_accuracy(model, val_loader, device, tasks)
        for i, task in enumerate(tasks):
            val_accuracies[i].append(val_accuracies_epoch[i])
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy ({task}): {val_accuracies_epoch[i]:.4f}")

        if epoch_val_loss < best_val_loss :
            file_path = os.path.join(model_save_dir, "best_model.pth")
            torch.save(model.state_dict(), file_path)
            logging.info(f"Best Validation Loss: {epoch_val_loss:.4f}")
            best_val_loss = epoch_val_loss

        if epoch % save_freq == 0:
            file_path = os.path.join(model_save_dir, f"model_{epoch}.pth")
            torch.save(model.state_dict(), file_path)

    # Plot and save training and validation loss and accuracy
    plt.figure(figsize=(12, 8))

    # Plot training and validation loss
    # plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(model_save_dir, 'loss_plots.png'))


    # Plot training and validation accuracy for each task
    for i, task in enumerate(tasks):
        plt.subplot(4, 2, i + 1)
        plt.plot(train_accuracies[i], label='Training Accuracy')
        plt.plot(val_accuracies[i], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Training and Validation Accuracy ({task})')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'training_validation_plots.png'))
    plt.show()


class OCT_Classifier(nn.Module):
    def __init__(self,num_classes=1) -> None:
        super(OCT_Classifier,self).__init__()

        self.base = models.resnet50(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        # x = self.sigmoid(x)

        return x

# Main function
def main():
    # Define hyperparameters
    batch_size = 128
    num_epochs = 20
    learning_rate = 0.0001
    logging.basicConfig(filename='training_logs_multiclass_models_resnet50_kfold.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Define paths and filenames
    csv_file = '/home/tejadhith/Project/OCT/Dataset/28-09-2023_download.csv'
    root_dir = '/home/tejadhith/Project/OCT/Dataset/segregated_28-sep-2023_kath'
    model_save_root_dir = '/home/tejadhith/Project/OCT/mutliclass_models_resnet50_kfold'

    # Create data loaders
    cross_val_loaders = create_k_fold_data_loaders(csv_file, root_dir, batch_size=batch_size)    
    # print("Length of Train Dataset : ",len(train_loader.dataset))
    # print("Length of Val Dataset : ",len(val_loader.dataset))
    # print("Length of Test Dataset : ",len(test_loader.dataset))

    # call the model
    # model = models_vit.__dict__['vit_large_patch16'](
    #     num_classes=8,
    #     drop_path_rate=0,
    #     global_pool=True,
    # )
    
    # # load RETFound weights
    # checkpoint = torch.load('RETFound_oct_weights.pth', map_location='cpu')
    # checkpoint_model = checkpoint['model']
    # state_dict = model.state_dict()
    # for k in ['head.weight', 'head.bias']:
    #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #         print(f"Removing key {k} from pretrained checkpoint")
    #         del checkpoint_model[k]

    # # interpolate position embedding
    # interpolate_pos_embed(model, checkpoint_model)

    # # # # load pre-trained model
    # msg = model.load_state_dict(checkpoint_model, strict=False)

    # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

    # # # manually initialize fc layer
    # trunc_normal_(model.head.weight, std=2e-5)

   
    # Define loss function and optimizer
    model = OCT_Classifier(num_classes=8)

    criterion = nn.CrossEntropyLoss()
    # criterion = custom_loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    tasks = ["foveal_scan","healthy","srf","irf","drusen","ped","hdots","hfoci"]
    
    final_accuracies = []
    for fold, (train_loader, val_loader, test_loader) in enumerate(cross_val_loaders, 1):
        logging.info(f"Fold {fold}:")
            # # # load pre-trained model
        # msg = model.load_state_dict(checkpoint_model, strict=False)
        # trunc_normal_(model.head.weight, std=2e-5)
        # Initial test accuracy
        test_accuracy_initial = calculate_accuracy(model, test_loader, device='cuda', tasks=tasks)
        for i, task in enumerate(tasks):
            logging.info(f"Initial Test Accuracy ({task}): {test_accuracy_initial[i]:.4f}")

        # Train the model
        model_save_dir = os.path.join(model_save_root_dir, f"fold_{fold}")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        logging.info(f"Saving models in {model_save_dir}")
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, tasks=tasks,
                    save_freq=5, model_save_dir=model_save_dir)

        # Final test accuracy
        test_accuracy_final = calculate_accuracy(model, test_loader, device='cuda', tasks=tasks)
        final_accuracies.append(test_accuracy_final)

        for i, task in enumerate(tasks):
            logging.info(f"Final Test Accuracy ({task}): {test_accuracy_final[i]:.4f}")

        ## Reinitializing
        model = OCT_Classifier(num_classes=8)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Print final accuracies
    logging.info("Final Accuracies:")
    for fold, acc in enumerate(final_accuracies, 1):
        logging.info(f"Fold {fold}: {', '.join([f'{task}: {acc[i]:.4f}' for i, task in enumerate(tasks)])}")

    # Calculate and print average accuracies for each task
    avg_accuracies = np.mean(np.array(final_accuracies), axis=0)
    logging.info("Average Accuracies:")
    for i, task in enumerate(tasks):
        logging.info(f"{task}: {avg_accuracies[i]:.4f}")


if __name__ == "__main__":
    main()
