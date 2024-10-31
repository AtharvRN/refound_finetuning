import torch
import torch.nn as nn
import torch.optim as optim
from data import create_data_loaders
import models_vit
from util.pos_embed import interpolate_pos_embed
from util.metrics import calculate_accuracy,calculate_metrics,calculate_combined_accuracy,calculate_combined_metrics
from util.plots import plot_confusion_matrix,plot_roc_curve,plot_confusion_matrices,plot_roc_curves,plot_precision_recall_curves
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
    parser.add_argument("--model_save_root", type=str, default=r"C:\Users\ARN162\Projects\OCT\Common_5", help="Root directory to save trained models")
    parser.add_argument("--model_name", type=str, default="resnet50", help="Name of the model to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--reset_model", type=bool, default=True, help="Learning rate for training")
    parser.add_argument("--task", type=str, default="healthy", help="Choose from  [healthy,foveal_scan,srf,irf,drusen,hdots,hfoci,ped]")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID : 0,1,2,3")
    parser.add_argument("--threshold", type=float, default=-1, help="User specified threshold (Valid for single task)")
    # parser.add_argument("--multiclass", type=bool, default=True, help="Whether binary classification or multiple binary classifications")
    parser.add_argument("--multitask", action="store_true", help="Whether binary classification or multiple binary classifications")


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
    print(args)
    # print("1")
    model_save_root_dir = os.path.join(args.model_save_root,args.model_name)
    if not os.path.exists(model_save_root_dir):
        print("Model Directory Does Not Exist")
        os.makedirs(model_save_root_dir)
        print("New Directory Created")
    else:
        print("Model Directory already exists")   
    batch_size = args.batch_size
  
    task = args.task
    device = "cuda:"+str(args.device_id)
    # Set up training log file

    # Set up evaluation log file
    if args.multitask == False:
        eval_log_file = os.path.join(model_save_root_dir, args.model_name + '_'+task+ '_evaluation_logs.txt')
        model_save_dir = os.path.join(model_save_root_dir, task)
        tasks = [task]
    else:
        eval_log_file = os.path.join(model_save_root_dir, args.model_name + '_evaluation_logs.txt')
        model_save_dir =model_save_root_dir
        tasks = ["foveal_scan","healthy","drusen","ped","hdots"]
    eval_log = open(eval_log_file, 'a')
    

    # Create data loaders
    _,_, test_loader = create_data_loaders(args.train_csv_file, args.train_root_dir, args.test_csv_file,args.test_root_dir, batch_size=batch_size,image_size=224)
    print("Length of Test Dataset : ",len(test_loader.dataset))

    logging.info(f"Length of Test Dataset : {len(test_loader.dataset)}")
    
    if args.model_name.lower() == "resnet50":

        logging.info("Pretrained Model Loaded")
        model = OCT_Classifier(num_classes=len(tasks))

    elif args.model_name.lower() == "retfound":
        model = models_vit.__dict__['vit_large_patch16'](
            num_classes=len(tasks),
            drop_path_rate=0,
            global_pool=True,
        )

    else:
        raise ValueError("model_name incorrect")
    
    
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
   
        
    logging.info("Saving models in %s ", model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)



    # Train the model
    best_model_path = os.path.join(model_save_dir,"best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    # Evaluate final performance
    # if args.multiclass == False:
    #     if args.threshold != -1:
    #         opt_threshold = args.threshold
    #     else:
        
    #         metrics = calculate_combined_metrics(model, test_loader, device=device, tasks=[task])
    #         opt_threshold = metrics[-1]
    #     conf_matrix, precision, recall, FPR,auc,fprs,tprs,thresholds,p4,_ = calculate_metrics(model, test_loader, device=device, task=task,threshold=opt_threshold)

    #     test_accuracy_final = calculate_accuracy(model, test_loader, device=device, task=task,threshold=opt_threshold)
    # else:
        
    metrics = calculate_combined_metrics(model, test_loader, device=device, tasks=tasks)
    # print(metrics)
    opt_threshold = {}
    for task in tasks:
        opt_threshold[task] = metrics[task]['log_metrics']['opt_thresh']
    metrics = calculate_combined_metrics(model, test_loader, device=device, tasks=tasks,threshold=opt_threshold)

    # test_accuracy_final = calculate_combined_accuracy(model, test_loader, device=device, tasks=tasks,threshold=opt_threshold)
        # print(metrics)

    # if args.multiclass == False:
    #     F1_score = 2*precision*recall/(precision+recall)
    #     # Write evaluation scores to the common evaluation log file
    #     eval_log.write(f"Task: {task}\n")
    #     eval_log.write(f"Threshold {opt_threshold}\n")
    #     eval_log.write(f"Test Accuracy : {test_accuracy_final}\n")
    #     eval_log.write(f"Precision : {precision}\n")
    #     eval_log.write(f"Recall : {recall}\n")
    #     eval_log.write(f"FPR : {FPR}\n")
    #     eval_log.write(f"AUC : {auc}\n ")
    #     eval_log.write(f"F1 Score : {F1_score}\n")
    #     eval_log.write(f"p4 metric : {p4}\n")
    #     eval_log.write("Confusion Matrix:\n")
    #     for i in range(conf_matrix.shape[0]):
    #         eval_log.write(" ".join(map(str, conf_matrix[i])) + "\n")
    #     eval_log.write("\n")
    #     eval_log.flush()
    # else:
    confusion_matrices = {}
    auc_values,auc_pr_values,fpr_values,tpr_values,threshold_values = {},{},{},{},{}
    precision_values,recall_values,threshold_values_pr = {},{},{}
    for task in tasks:
        eval_log.write(f"Task: {task}\n")

        eval_log.write(f"Threshold {opt_threshold[task]}\n")
        # eval_log.write(f"Test Accuracy : {test_accuracy_final[task]}\n")

        for key in metrics[task]['log_metrics']:
            if key == 'conf_matrix':
                conf_matrix =  metrics[task]['log_metrics'][key]
                for i in range(conf_matrix.shape[0]):
                    eval_log.write(" ".join(map(str, conf_matrix[i])) + "\n")   
            else :
                eval_log.write(f"{key} : {metrics[task]['log_metrics'][key]}\n")
        # eval_log.write(f"Precision : {metrics[task]['precision']}\n")
        # eval_log.write(f"Recall : {metrics[task]['recall']}\n")
        # eval_log.write(f"FPR : {metrics[task]['FPR']}\n")
        # eval_log.write(f"AUC ROC: {metrics[task]['auc_roc']}\n ")
        # eval_log.write(f"AUC PR : {metrics[task]['auc_pr']}\n")
        # eval_log.write(f"F1 Score : {metrics[task]['F1']}\n")
        # eval_log.write(f"p4 metric : {p4[task]}\n")
        # eval_log.write("Confusion Matrix:\n")
        # conf_matrix = metrics[task]['conf_matrix']
        # for i in range(conf_matrix.shape[0]):
        #     eval_log.write(" ".join(map(str, conf_matrix[i])) + "\n")

        confusion_matrices[task] = conf_matrix
        auc_values[task] = metrics[task]['log_metrics']['auc_roc']
        auc_pr_values[task] = metrics[task]['log_metrics']['auc_pr']
        fpr_values[task] =metrics[task]['plot_metrics']['fprs']
        tpr_values[task] = metrics[task]['plot_metrics']['tprs']
        threshold_values[task] =metrics[task]['plot_metrics']['thresholds']
        precision_values[task] = metrics[task]['plot_metrics']['precisions'][:-1]
        recall_values[task] = metrics[task]['plot_metrics']['recalls'][:-1]
        threshold_values_pr[task] = metrics[task]['plot_metrics']['threshold_values']
        eval_log.write("\n")
        eval_log.flush()
        

    # Close the evaluation log file
    eval_log.close()
    figures_dir = os.path.join(model_save_root_dir,"figures_optimum_threshold")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    # print(figures_dir)
    
    # if args.multiclass == False:
    #     cf_filename = os.path.join(figures_dir,"confusion_matrix_"+task+"_"+str(opt_threshold)+".png")
    #     plot_confusion_matrix(conf_matrix[task], classes=["Negative", "Positive"], title=f'Confusion matrix for {task}',save_path=cf_filename)
    #     roc_filename = os.path.join(figures_dir,"roc_"+task+".png")
    #     plot_roc_curve(auc_val, fprs, tprs, thresholds,save_path=roc_filename)

    # else:

    cf_filename = os.path.join(figures_dir,"confusion_matrix.png")
    plot_confusion_matrices(confusion_matrices,tasks,cf_filename)
    roc_filename = os.path.join(figures_dir,"roc_curves")
    if not os.path.exists(roc_filename):
        os.makedirs(roc_filename)
    plot_roc_curves(auc_values, fpr_values, tpr_values,threshold_values, roc_filename)
    pr_filename = os.path.join(figures_dir,"pr_curves")
    if not os.path.exists(pr_filename):
        os.makedirs(pr_filename)
    plot_precision_recall_curves(auc_pr_values,precision_values, recall_values, threshold_values_pr, pr_filename)

if __name__ == "__main__":
    main()
