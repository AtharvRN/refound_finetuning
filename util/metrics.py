import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve,precision_recall_curve,average_precision_score

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
def calculate_combined_accuracy(model, data_loader, device='cuda', tasks= ["foveal_scan","healthy","srf","irf","drusen","hdots","hfoci","ped"], threshold=None):
    model = model.to(device)
    model.eval()
    
    num_tasks = len(tasks)
    correct = [0] * num_tasks
    total = [0] * num_tasks
    accuracies = {}
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = {key: value.to(device) for key, value in labels.items()}  # Move all labels to device
            labels = torch.stack([labels[key] for key in labels], dim=1)
    
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            for i,task in enumerate(tasks):
                if threshold is None:
                    predictions = (outputs[:, i] > 0.5).float()
                else:
                    predictions = (outputs[:, i] > threshold[task]).float()
                correct[i] += (predictions == labels[:,i]).sum().item()
                total[i] += labels.size(0)

                accuracies[tasks[i]] = correct[i] / total[i]
    return accuracies

def optimal_threshold(fpr,tpr,thresholds):
    index = np.argmax((1-fpr)**2 +tpr**2)
    return thresholds[index]

def calculate_metrics(model, data_loader, device='cuda', task='healthy', threshold=0.5):
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels[task].unsqueeze(1).float().to(device)
            outputs = model(inputs)
            # if multiclass
            outputs = torch.sigmoid(outputs)
            predictions = (outputs > threshold).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    FPR = FP / (FP + TN)
    # Precision, Recall, TPR, FPR
    # true_positive = np.sum((y_true == 1) & (y_pred == 1))
    # false_positive = np.sum((y_true == 0) & (y_pred == 1))
    # false_negative = np.sum((y_true == 1) & (y_pred == 0))
    # true_negative = np.sum((y_true == 0) & (y_pred == 0))
    # precision = true_positive / (true_positive + false_positive)
    # recall = true_positive / (true_positive + false_negative)
    # FPR = false_positive / (false_positive + true_negative)

    # AUC-ROC
    auc_roc = roc_auc_score(y_true, y_scores)
    # p4_metric = (4*true_positive*true_negative)/(4*true_positive*true_negative+(true_negative+true_positive)*(false_positive+false_negative))
    # ROC Curve
    fprs, tprs, thresholds= roc_curve(y_true, y_scores)
    opt_thresh = optimal_threshold(fprs,tprs,thresholds)
    fprs, tprs, thresholds = roc_curve(y_t,y_s)
    opt_thresh = optimal_threshold(fprs,tprs,thresholds)
    precisions, recalls, threshold_values = precision_recall_curve(y_t, y_s)
    auc_pr = average_precision_score(y_t, y_s)
    max_precision, max_precision_threshold, max_recall, max_recall_threshold = max_precision_recall_thresholds(precisions,recalls,threshold_values)

    return conf_matrix, precision, recall, FPR, auc_roc,fprs,tprs,thresholds,p4_metric,opt_thresh

def max_precision_recall_thresholds(precisions,recalls,thresholds):

    precisions = precisions[:-1]
    recalls = recalls[:-1]
    thresholds = thresholds

    max_precision_idx = precisions.argmax()
    max_recall_idx = recalls.argmax()

    max_precision = precisions[max_precision_idx]
    max_precision_threshold = thresholds[max_precision_idx]

    max_recall = recalls[max_recall_idx]
    max_recall_threshold = thresholds[max_recall_idx]

    return max_precision, max_precision_threshold, max_recall, max_recall_threshold


def calculate_combined_metrics(model, data_loader, device='cuda', tasks=["foveal_scan", "healthy", "srf", "irf", "drusen", "hdots", "hfoci", "ped"], threshold=None):
    model = model.to(device)
    model.eval()

    metrics = {}
    y_true ={}
    y_pred = {}
    y_score = {}
    for task in tasks:
        y_true[task] = []
        y_score[task] = []
        y_pred[task] = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            if len(tasks) == 1:
                labels = labels[task].unsqueeze(1).float().to(device)
            else:
                labels = {key: value.to(device) for key, value in labels.items()}  # Move all labels to device
                labels = torch.stack([labels[key] for key in labels], dim=1)
            # print(labels.shape)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)

            # print(outputs.shape)
           
            for i, task in enumerate(tasks):
                if threshold is None:
                    predictions = (outputs[:, i] > 0.5).int()
                else:
                    # print(task,threshold[task])
                    predictions = (outputs[:, i] > threshold[task]).int()
                    # print(task,outputs,predictions,labels)
                correct = (predictions == labels[:, i]).sum().item()
                # print(correct)
                total = labels.size(0)
                # print(total)
                y_true[task].extend(labels[:,i].cpu().numpy())
                y_pred[task].extend(predictions.cpu().numpy())
                # print(outputs[:i].shape)
                y_score[task].extend(outputs[:,i].cpu().numpy())
                # Confusion Matrix
                # print(len(y_true[task]))
                # print(len(y_score[task]))
            # print(y_true)
            # print(y_pred)
            # print(y_score)
        for task in tasks:
            y_t = y_true[task]
            # print(y_t)
            y_p = y_pred[task]
            y_s = y_score[task]
            # print(y_s)
            # print(len(y_t))
            # print(len(y_p))
            # print(len(y_s))

            conf_matrix = confusion_matrix(y_t, y_p)
            # print(y_t,y_p)
            # print(len(y_t))
            # Precision, Recall, FPR
            # true_positive = np.sum((y_t == 1) & (y_p == 1))
            # false_positive = np.sum((y_t == 0) & (y_p == 1))
            # false_negative = np.sum((y_t == 1) & (y_p == 0))
            # true_negative = np.sum((y_t == 0) & (y_p == 0))
            # print(true_positive,false_positive,false_negative,true_negative)
            # precision = true_positive / (true_positive + false_positive)
            # recall = true_positive / (true_positive + false_negative)
            # FPR = false_positive / (false_positive + true_negative)
            TN, FP, FN, TP = conf_matrix.ravel()

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            FPR = FP / (FP + TN)
 
            F1_score = 2*precision*recall/(precision+recall)
            # AUC-ROC
            auc_roc = roc_auc_score(y_t, y_s)
            

            # ROC Curve
            fprs, tprs, thresholds = roc_curve(y_t,y_s)
            opt_thresh = optimal_threshold(fprs,tprs,thresholds)
            precisions, recalls, threshold_values = precision_recall_curve(y_t, y_s)
            auc_pr = average_precision_score(y_t, y_s)
            max_precision, max_precision_threshold, max_recall, max_recall_threshold = max_precision_recall_thresholds(precisions,recalls,threshold_values)

            # print(precisions)
            # print(task,thresholds,threshold_values)
            # Store metrics in dictionary
            if task not in metrics:
                metrics[task] = []

            metrics[task]= {
                'log_metrics':{
                'accuracy': correct / total,
                'conf_matrix': conf_matrix,
                'precision': precision,
                'recall': recall,
                'FPR': FPR,
                'F1' : F1_score,
                'auc_roc': auc_roc,
                'auc_pr':auc_pr,
                'p_max': max_precision,
                'p_max_thresh':max_precision_threshold,
                'r_max': max_recall,
                'r_max_thresh':max_recall_threshold,
                'opt_thresh':opt_thresh,
                },
                'plot_metrics':{
                'fprs': fprs,
                'tprs': tprs,
                'thresholds': thresholds,
                'precisions': precisions,
                'recalls': recalls,
                'threshold_values': threshold_values,
                }
                
            }

    return metrics


