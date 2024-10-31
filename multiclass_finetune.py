import torch
import torch.nn as nn
import torch.optim as optim
from data import create_data_loaders
import models_vit
from util.pos_embed import interpolate_pos_embed
from util.metrics import calculate_combined_accuracy as calculate_accuracy
from util.metrics import calculate_combined_metrics as calculate_metrics
from util.plots import plot_confusion_matrices,plot_roc_curves
from timm.models.layers import trunc_normal_
import datetime
import os
import logging
from torchvision import models
import argparse
import matplotlib.pyplot as plt
# Configure logger to write to a file
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train OCT classifier model")
    parser.add_argument("--train_csv_file", type=str, default="C:/Users/ARN162/Projects/OCT/Dataset/28-09-2023_download.csv", help="Path to train CSV file")
    parser.add_argument("--test_csv_file", type=str, default=r"C:\Users\ARN162\Projects\OCT\Dataset\16-03-2024_download2.csv", help="Path to test CSV file")
    parser.add_argument("--train_root_dir", type=str, default=r"C:\Users\ARN162\Projects\OCT\Dataset\segregated_28-sep-2023_kath", help="Root directory of train dataset")
    parser.add_argument("--test_root_dir", type=str, default=r"C:\Users\ARN162\Projects\OCT\Dataset\stavan_images_march25", help="Root directory of test dataset")
    parser.add_argument("--model_save_root", type=str, default=r"C:\Users\ARN162\Projects\OCT\Common_5", help="Root directory to save trained models")
    parser.add_argument("--model_name", type=str, default="resnet50", help="Name of the model to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--reset_model", type=bool, default=True, help="Learning rate for training")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID : 0,1,2,3")

    return parser.parse_args()


def custom_loss(output, target):
    # Assuming output and target are both of shape (batch_size, 8)
    loss = 0.0
    criterion = nn.BCELoss()  # Sum the losses across the batch
    for i in range(output.size(1)):  # Iterate over each dimension of the output
        loss += criterion(output[:, i], target[:, i].float()) 
    return loss

def train_model(model, train_loader, val_loader, optimizer, num_epochs=10, device='cuda', tasks=["foveal_scan", "healthy", "srf", "irf", "drusen", "hdots", "hfoci", "ped"], save_freq=5, model_save_dir=""):
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

        train_accuracy_epoch = calculate_accuracy(model, train_loader, device, tasks)
        # train_accuracies[i].append(train_accuracy_epoch)
        for i, task in enumerate(tasks):
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Accuracy ({task}): {train_accuracy_epoch[task]:.4f}")

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
        val_accuracy_epoch = calculate_accuracy(model, val_loader, device, tasks)

        for i, task in enumerate(tasks):
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy ({task}): {val_accuracy_epoch[task]:.4f}")

        if epoch_val_loss < best_val_loss :
            file_path = os.path.join(model_save_dir, "best_model.pth")
            torch.save(model.state_dict(), file_path)
            logging.info(f"Best Validation Loss: {epoch_val_loss:.4f}")
            best_val_loss = epoch_val_loss

        if epoch % save_freq == 0:
            file_path = os.path.join(model_save_dir, f"model_{epoch}.pth")
            torch.save(model.state_dict(), file_path)

    # Plot and save training and validation loss and accuracy
    # plt.figure(figsize=(12, 8))

    # # Plot training and validation loss
    # # plt.subplot(2, 2, 1)
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.savefig(os.path.join(model_save_dir, 'loss_plots.png'))


    # # Plot training and validation accuracy for each task
    # for i, task in enumerate(tasks):
    #     plt.subplot(4, 2, i + 1)
    #     plt.plot(train_accuracies[i], label='Training Accuracy')
    #     plt.plot(val_accuracies[i], label='Validation Accuracy')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    #     plt.title(f'Training and Validation Accuracy ({task})')
    #     plt.legend()

    # plt.tight_layout()
    # plt.savefig(os.path.join(model_save_dir, 'training_validation_plots.png'))
    # plt.show()


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
def init_retfound(model):
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

    # # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)

    return model
# Main function
def main():
    # Define hyperparameters
    args = parse_arguments()
    # print("1")
    model_save_root_dir = os.path.join(args.model_save_root,args.model_name)
    print(model_save_root_dir)
    if not os.path.exists(model_save_root_dir):
        print("Model Directory Does Not Exist")
        os.makedirs(model_save_root_dir)
        print("New Directory Created")
    else:
        print("Model Directory already exists")   
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    device = "cuda:"+str(args.device_id)
    # Set up training log file
    training_log_file = os.path.join(model_save_root_dir, args.model_name + '_training_logs.txt')
    logging.basicConfig(filename=training_log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set up evaluation log file
    eval_log_file = os.path.join(model_save_root_dir, args.model_name + '_evaluation_logs.txt')
    eval_log = open(eval_log_file, 'a')

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(args.train_csv_file, args.train_root_dir, args.test_csv_file,args.test_root_dir, batch_size=batch_size,image_size=224)
    print("Length of Train Dataset : ",len(train_loader.dataset))
    print("Length of Val Dataset : ",len(val_loader.dataset))
    print("Length of Test Dataset : ",len(test_loader.dataset))

    logging.info(f"Length of Train Dataset : {len(train_loader.dataset)}")
    logging.info(f"Length of Val Dataset : {len(val_loader.dataset)}")
    logging.info(f"Length of Test Dataset : {len(test_loader.dataset)}")
    
    tasks = ["foveal_scan","healthy","drusen","ped","hdots"]

    if args.model_name.lower() == "resnet50":

        logging.info("Pretrained Model Loaded")
        model = OCT_Classifier(num_classes=len(tasks))

    elif args.model_name.lower() == "retfound":
        model = models_vit.__dict__['vit_large_patch16'](
            num_classes=len(tasks),
            drop_path_rate=0,
            global_pool=True,
        )
        model = init_retfound(model)

    else:
        raise ValueError("model_name incorrect")

    # tasks = ["foveal_scan","healthy"]
    # Define loss function and optimizer
    # model = OCT_Classifier(num_classes=len(tasks))

    # criterion = nn.CrossEntropyLoss()
    # criterion = custom_loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    model_save_dir = model_save_root_dir
    logging.info("Saving models in %s ",model_save_dir)
    # for num_epoch in num_epochs:
    
    test_accuracy_initial = calculate_accuracy(model, test_loader, device,tasks)
    metrics = calculate_metrics(model, test_loader, device,tasks)
    for task in tasks:
        logging.info(f"Task: {task}")   
        logging.info("Test Accuracy (Initial) : %s", test_accuracy_initial[task])
        logging.info("Precision (Initial) : %s", metrics[task]["precision"])
        logging.info("Recall (Initial) : %s", metrics[task]["recall"])
        logging.info("FPR (Initial) : %s", metrics[task]["FPR"])
        # Write evaluation scores to the common evaluation log file
        eval_log.write(f"Task: {task}\n")
        eval_log.write(f"Test Accuracy (Initial): {test_accuracy_initial[task]}\n")
        eval_log.write(f"Precision (Initial): {metrics[task]['precision']}\n")
        eval_log.write(f"Recall (Initial): {metrics[task]['recall']}\n")
        eval_log.write(f"FPR (Initial): {metrics[task]['FPR']}\n")
        eval_log.flush()
        

    train_model(model, train_loader, val_loader,optimizer, num_epochs =num_epochs,tasks=tasks,save_freq =5,model_save_dir=model_save_dir)
    
    best_model_path = os.path.join(model_save_dir,"best_model.pth")
    model.load_state_dict(torch.load(best_model_path))

    confusion_matrices = {}
    auc_values,fpr_values,tpr_values,threshold_values = {},{},{},{}

    # for i, task in enumerate(tasks):
    test_accuracy_final = calculate_accuracy(model, test_loader, device,tasks)
    metrics = calculate_metrics(model, test_loader, device, tasks)
    # print(metrics)
    for task in tasks:
        # print(confusion_matrices)
        logging.info("Test Accuracy (Final) : %s", test_accuracy_final[task])
        logging.info("Precision (Final) : %s", metrics[task]["precision"])
        logging.info("Recall (Final) : %s", metrics[task]["recall"])
        logging.info("FPR (Final) : %s", metrics[task]["FPR"])
        logging.info("AUC-ROC (Final) : %s", metrics[task]["auc_roc"])

        eval_log.write(f"Test Accuracy (Final): {test_accuracy_final[task]}\n")
        eval_log.write(f"Precision (Final): {metrics[task]['precision']}\n")
        eval_log.write(f"Recall (Final): {metrics[task]['recall']}\n")
        eval_log.write(f"FPR (Final): {metrics[task]['FPR']}\n")
        eval_log.write(f"AUC-ROC (Final): {metrics[task]['auc_roc']}\n")
        eval_log.write(f"Optimum Threshold: {metrics[task]['opt_thresh']}\n")
        confusion_matrices[task] = metrics[task]['conf_matrix']
        auc_values[task] = metrics[task]['auc_roc']
        fpr_values[task] =metrics[task]['fprs']
        tpr_values[task] = metrics[task]['tprs']
        threshold_values[task] =metrics[task]['thresholds']
        eval_log.write("Confusion Matrix:\n")
        for i in range(metrics[task]['conf_matrix'].shape[0]):
            eval_log.write(" ".join(map(str, metrics[task]['conf_matrix'][i])) + "\n")
        eval_log.write("\n")
        eval_log.flush()

    eval_log.close()
    figures_dir = os.path.join(model_save_root_dir,"figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    # print(figures_dir)
    cf_filename = os.path.join(figures_dir,"confusion_matrix.png")
    plot_confusion_matrices(confusion_matrices,tasks,cf_filename)
    roc_filename = os.path.join(figures_dir)
    plot_roc_curves(auc_values, fpr_values, tpr_values,threshold_values, roc_filename)


if __name__ == "__main__":
    main()
