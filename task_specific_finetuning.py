import torch
import torch.nn as nn
import torch.optim as optim
from data import create_data_loaders
import models_vit
from models_vit import VIT_classifier
from util.pos_embed import interpolate_pos_embed
from util.metrics import calculate_accuracy,calculate_metrics
from util.plots import plot_confusion_matrix,plot_roc_curve
from timm.models.layers import trunc_normal_
import os
import logging
from torchvision import models
import itertools
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train OCT classifier model")
    parser.add_argument("--train_csv_file", type=str, default="C:/Users/ARN162/Projects/OCT/Dataset/28-09-2023_download.csv", help="Path to train CSV file")
    parser.add_argument("--test_csv_file", type=str, default=r"C:\Users\ARN162\Projects\OCT\Dataset\16-03-2024_download2.csv", help="Path to test CSV file")
    parser.add_argument("--train_root_dir", type=str, default=r"C:\Users\ARN162\Projects\OCT\Dataset\preprocessed_train", help="Root directory of train dataset")
    parser.add_argument("--test_root_dir", type=str, default=r"C:\Users\ARN162\Projects\OCT\Dataset\preprocessed_test", help="Root directory of test dataset")
    parser.add_argument("--model_save_root", type=str, default=r"C:\Users\ARN162\Projects\OCT\Separate_specific", help="Root directory to save trained models")
    parser.add_argument("--model_name", type=str, default="resnet50", help="Name of the model to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--reset_model", type=bool, default=True, help="Learning rate for training")
    parser.add_argument("--task", type=str, default="healthy", help="Choose from  [healthy,foveal_scan,srf,irf,drusen,hdots,hfoci,ped]")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID : 0,1,2,3")
    parser.add_argument("--pos_weight", type=float, default=1, help="Weightage given to posiive class")

    # parser.add_argument("--lr", type=float, default="hea, help="Choose from  [healthy,foveal_scan,srf,irf,drusen,hdots,hfoci,ped]")

    return parser.parse_args()

# Function for training the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda:0',task='healthy',save_freq =5,model_save_dir=""):
    best_val_acc = 0
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct,total = 0,0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels[task].unsqueeze(1).float().to(device)
            # print(inputs[0])
            # print(inputs[1])
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            predictions = (torch.sigmoid(outputs) > 0.5)
            # print(predictions)
            # print(labels)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            # print(correct,total)
            # print(outputs)
            # print(torch.sigmoid(outputs))
            # outputs = torch.sigmoid(outputs)
            
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")
        train_accuracy = correct/total
        # Calculate training accuracy
        # train_accuracy = calculate_accuracy(model, train_loader, device,task)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Accuracy: {train_accuracy:.4f}")

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels[task].unsqueeze(1).float().to(device)
                outputs = model(inputs)
                
                # outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)
        logging.info(f"Validation Loss: {val_loss:.4f}")

        # Calculate validation accuracy
        val_accuracy = calculate_accuracy(model, val_loader, device,task) 

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            file_path = os.path.join(model_save_dir,"best_model.pth")
            torch.save(model.state_dict(),file_path)
            logging.info(f"Best Validation Accuracy: {val_accuracy:.4f}")
        else:
            logging.info(f"Validation Accuracy: {val_accuracy:.4f}")
        if epoch%save_freq == 0:
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
    task = args.task
    device = "cuda:"+str(args.device_id)
    # Set up training log file
    training_log_file = os.path.join(model_save_root_dir, args.model_name + '_'+task+ '_training_logs.txt')
    logging.basicConfig(filename=training_log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set up evaluation log file
    eval_log_file = os.path.join(model_save_root_dir, args.model_name + '_'+task+ '_evaluation_logs.txt')
    eval_log = open(eval_log_file, 'a')
    

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(args.train_csv_file, args.train_root_dir, args.test_csv_file,args.test_root_dir, batch_size=batch_size,image_size=224)
    print("Length of Train Dataset : ",len(train_loader.dataset))
    print("Length of Val Dataset : ",len(val_loader.dataset))
    print("Length of Test Dataset : ",len(test_loader.dataset))

    logging.info(f"Length of Train Dataset : {len(train_loader.dataset)}")
    logging.info(f"Length of Val Dataset : {len(val_loader.dataset)}")
    logging.info(f"Length of Test Dataset : {len(test_loader.dataset)}")
    
    if args.model_name.lower() == "resnet50":

        logging.info("Pretrained Model Loaded")
        model = OCT_Classifier(num_classes=1)

    elif args.model_name.lower() == "retfound":
        vit_model = models_vit.__dict__['vit_large_patch16'](
            num_classes=1,
            drop_path_rate=0,
            global_pool=True,
        )
        model = init_retfound(vit_model)
        # model = VIT_classifier(vit_model)

    else:
        raise ValueError("model_name incorrect")
    
    
    pos_weight = args.pos_weight
    logging.info(f"pos_weight = {pos_weight}")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # if args.reset_model:
            # if args.model_name == "resnet50":
            #     model.module.reset_parameters()
            # elif args.model_name == "retfound":
            #     model = init_retfound(model)
        # if torch.cuda.device_count() > 1:  # Check for multiple GPUs
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model)
        # # print(model)
        # Train the model
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    model_save_dir = os.path.join(model_save_root_dir, task)
    logging.info("Saving models in %s ", model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # Evaluate initial performance
    test_accuracy_initial = calculate_accuracy(model, test_loader, device=device, task=task)
    conf_matrix, precision, recall,FPR,auc,fprs,tprs,thresholds,_ = calculate_metrics(model, test_loader, device=device, task=task)
    # print(confusion_matrices)
    logging.info("Test Accuracy (Initial) : %s", test_accuracy_initial)
    logging.info("Precision (Initial) : %s", precision)
    logging.info("Recall (Initial) : %s", recall)
    logging.info("FPR (Initial) : %s", FPR)
    eval_log.write(f"Test Accuracy (Initial): {test_accuracy_initial}\n")
    eval_log.write(f"Precision (Initial): {precision}\n")
    eval_log.write(f"Recall (Initial): {recall}\n")
    eval_log.write(f"FPR (Initial): {FPR}\n")

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs,device =device , task=task, save_freq=5, model_save_dir=model_save_dir)
    best_model_path = os.path.join(model_save_dir,"best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    # Evaluate final performance
    test_accuracy_final = calculate_accuracy(model, test_loader, device=device, task=task)
    conf_matrix, precision, recall, FPR,auc,fprs,tprs,thresholds,_ = calculate_metrics(model, test_loader, device=device, task=task)
    
    # print(confusion_matrices)
    logging.info(f"Task: {task}\n")
    logging.info("Test Accuracy (Final) : %s", test_accuracy_final)
    logging.info("Precision (Final) : %s", precision)
    logging.info("Recall (Final) : %s", recall)
    logging.info("FPR (Final) : %s", FPR)

    # Write evaluation scores to the common evaluation log file
    eval_log.write(f"Task: {task}\n")
    eval_log.write(f"Test Accuracy (Final): {test_accuracy_final}\n")
    eval_log.write(f"Precision (Final): {precision}\n")
    eval_log.write(f"Recall (Final): {recall}\n")
    eval_log.write(f"FPR (Final): {FPR}\n")
    eval_log.write(f"AUC : {auc}")
    eval_log.write("Confusion Matrix:\n")
    for i in range(conf_matrix.shape[0]):
        eval_log.write(" ".join(map(str, conf_matrix[i])) + "\n")
    eval_log.write("\n")
    eval_log.flush()
    
    # Close the evaluation log file
    eval_log.close()
    figures_dir = os.path.join(model_save_root_dir,"figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    # print(figures_dir)
    cf_filename = os.path.join(figures_dir,"confusion_matrix_"+task+".png")

    plot_confusion_matrix(conf_matrix, classes=["Negative", "Positive"], title=f'Confusion matrix for {task}',save_path=cf_filename)
    roc_filename = os.path.join(figures_dir,"roc_"+task+".png")
    plot_roc_curve(auc, fprs, tprs, thresholds,save_path=roc_filename)


if __name__ == "__main__":
    main()
