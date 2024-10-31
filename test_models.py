import torch
import torch.nn as nn
import torch.optim as optim
from data import create_data_loaders
import models_vit
from util.pos_embed import interpolate_pos_embed
from util.metrics import calculate_accuracy,calculate_metrics
from util.plots import plot_confusion_matrices,plot_roc_curves
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
    parser.add_argument("--train_root_dir", type=str, default=r"C:\Users\ARN162\Projects\OCT\Dataset\segregated_28-sep-2023_kath", help="Root directory of train dataset")
    parser.add_argument("--test_root_dir", type=str, default=r"C:\Users\ARN162\Projects\OCT\Dataset\preprocessed_test", help="Root directory of test dataset")
    parser.add_argument("--model_save_root", type=str, default=r"C:\Users\ARN162\Projects\OCT\separate2", help="Root directory to save trained models")
    parser.add_argument("--model_name", type=str, default="resnet50", help="Name of the model to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--reset_model", type=bool, default=True, help="Learning rate for training")

    return parser.parse_args()

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

    # Set up evaluation log file
    eval_log_file = os.path.join(model_save_root_dir, args.model_name + '_evaluation_logs_thresholds.txt')
    eval_log = open(eval_log_file, 'a')
    

    # Create data loaders
    _, _, test_loader = create_data_loaders(args.train_csv_file, args.train_root_dir, args.test_csv_file,args.test_root_dir, batch_size=batch_size,image_size=224)
    
    print("Length of Test Dataset : ",len(test_loader.dataset))
    
    if args.model_name.lower() == "resnet50":

        logging.info("Pretrained Model Loaded")
        model = OCT_Classifier(num_classes=1)

    elif args.model_name.lower() == "retfound":
        model = models_vit.__dict__['vit_large_patch16'](
            num_classes=1,
            drop_path_rate=0,
            global_pool=True,
        )

    else:
        raise ValueError("model_name incorrect")
    
    
    tasks = ["healthy","foveal_scan","srf","irf","drusen","hdots","hfoci","ped"]
    est_threshold = {
        "healthy" : 0.12,
        "foveal_scan":0.05,
        "srf" : 0.08,
        "irf":0.35,
        "drusen":0.23,
        "hdots":0.57,
        "hfoci":0.02,
        "ped" :0.05
    }
    # tasks = ["ped"]
    confusion_matrices = {}
    auc_values,fpr_values,tpr_values,threshold_values = {},{},{},{}
    for task in tasks:

        model_save_dir = os.path.join(model_save_root_dir, task)

        best_model_path = os.path.join(model_save_dir,"best_model.pth")
        model.load_state_dict(torch.load(best_model_path))
        # Evaluate final performance
        conf_matrix, precision, recall, FPR,auc,fprs,tprs,thresholds,opt_thresh = calculate_metrics(model, test_loader, device='cuda', task=task)
        
        test_accuracy_final = calculate_accuracy(model, test_loader, device='cuda', task=task,threshold=opt_thresh)
        confusion_matrices[task] = conf_matrix
        auc_values[task] = auc
        fpr_values[task] = fprs
        tpr_values[task] = tprs
        threshold_values[task] = thresholds
        eval_log.write(f"Task: {task}\n")
        eval_log.write(f"Optimal Threshold : {opt_thresh}\n")
        eval_log.write(f"Test Accuracy : {test_accuracy_final}\n")
        eval_log.write(f"Precision : {precision}\n")
        eval_log.write(f"Recall : {recall}\n")
        eval_log.write(f"FPR : {FPR}\n")
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
    cf_filename = os.path.join(figures_dir,"confusion_matrix.png")
    plot_confusion_matrices(confusion_matrices,tasks,cf_filename)
    roc_filename = os.path.join(figures_dir)
    plot_roc_curves(auc_values, fpr_values, tpr_values,threshold_values, roc_filename)


if __name__ == "__main__":
    main()
