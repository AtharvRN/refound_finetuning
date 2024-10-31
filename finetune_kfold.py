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
import numpy as np
# Configure logger to write to a file

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
            # print(outputs)
            outputs = torch.sigmoid(outputs)
            # print(outputs)
            predictions = (outputs > threshold).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

# Function for training the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda',task='healthy',save_freq =5,model_save_dir=""):
    
    model.to(device)
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels[task].unsqueeze(1).float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)
            # print(outputs)
            # print(labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

        # Calculate training accuracy
        train_accuracy = calculate_accuracy(model, train_loader, device,task)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Accuracy: {train_accuracy:.4f}")

        # Validate the model
        # model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels[task].unsqueeze(1).float().to(device)
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
        logging.info(f"Validation Loss: {val_loss:.4f}")

        # Calculate validation accuracy
        val_accuracy = calculate_accuracy(model, val_loader, device,task) 

        if val_accuracy < best_val_acc:
            best_val_acc = val_accuracy
            file_path = os.path.join(model_save_dir,"best_model.pth")
            torch.save(model.state_dict(),file_path)
            logging.info(f"Best Validation Accuracy: {val_accuracy:.4f}")
        else:
            logging.info(f"Validation Accuracy: {val_accuracy:.4f}")
        if epoch%save_freq:
            file_path = os.path.join(model_save_dir,"model_"+str(epoch)+".pth")
            torch.save(model.state_dict(),file_path)
    return best_val_acc

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
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    logging.basicConfig(filename='training_logs_retfound_kfold.txt', level=logging.INFO)

    # Define paths and filenames
    csv_file = '/home/tejadhith/Project/OCT/Dataset/28-09-2023_download.csv'
    root_dir = '/home/tejadhith/Project/OCT/Dataset/segregated_28-sep-2023_kath'
    model_save_root_dir = '/home/tejadhith/Project/OCT/models_retfound_kfold'

    # Create data loaders
    cross_val_loaders = create_k_fold_data_loaders(csv_file, root_dir, batch_size=batch_size)    
    # print("Length of Train Dataset : ",len(train_loader.dataset))
    # print("Length of Val Dataset : ",len(val_loader.dataset))
    # print("Length of Test Dataset : ",len(test_loader.dataset))

    # resnet_model = ResNet34()
    # resnet_model = load_checkpoint(resnet_model,"/raid/ee19resch01008/Atharv/Project/OCT/resnet/ckpt130.pt")
    # # Initialize model
    # logging.info("Pretrained Model Loaded")
    # resnet_base = resnet_model.resnet
    # model = OCT_Classifier(resnet_base)

    # call the model
    model = models_vit.__dict__['vit_large_patch16'](
        num_classes=1,
        drop_path_rate=0,
        global_pool=True,
    )
    
    # load RETFound weights
    checkpoint = torch.load('RETFound_oct_weights.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # # # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    trunc_normal_(model.head.weight, std=2e-5)

    # # manually initialize fc layer
    # print(model)
    # print("Before optimization:")
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print("yes")
    #         print(name, param.data)
    # Define loss function and optimizer
    # model = models.resnet50(pretrained=True,)
    # model = OCT_Classifier(num_classes=6)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    tasks = ["foveal_scan","healthy","srf","irf","drusen","ped","hdots","hfoci"]
    # tasks = ["ped"]
    avg_accuracies = []

    for task in tasks:
    # Train the model
        logging.info("Current Task : %s ",task)
        accuracies = []

        for fold, (train_loader, val_loader, test_loader) in enumerate(cross_val_loaders, 1):
            msg = model.load_state_dict(checkpoint_model, strict=False)
            trunc_normal_(model.head.weight, std=2e-5)

            logging.info(f"Fold {fold}:")
            model_save_dir = os.path.join(model_save_root_dir,task)
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            model_save_fold_dir = os.path.join(model_save_dir,f"fold_{fold}")
            if not os.path.exists(model_save_fold_dir):
                os.makedirs(model_save_fold_dir)
            logging.info("Saving models in %s ",model_save_fold_dir)
            test_accuracy_initial = calculate_accuracy(model, test_loader, device='cuda',task=task)
            logging.info("Test Accuracy (Initial) : %s",test_accuracy_initial)
            train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs =num_epochs,task=task,save_freq =5,model_save_dir=model_save_fold_dir)
            test_accuracy_final = calculate_accuracy(model, test_loader, device='cuda',task=task)
            accuracies.append(test_accuracy_final)
            logging.info("Test Accuracy (Final) : %s",test_accuracy_final)
        avg_accuracy = np.mean(accuracies)
        logging.info(f"Average Accuracy for {task} task : {avg_accuracy}")

        avg_accuracies.append(avg_accuracy)

    logging.info("Average Accuracies:")
    for i, task in enumerate(tasks):
        logging.info(f"{task}: {avg_accuracies[i]:.4f}")

if __name__ == "__main__":
    main()
