import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
def plot_confusion_matrix(cm, classes, ax=None, title='Confusion matrix', cmap=plt.cm.Blues,save_path="conf.png"):
    """
    This function plots the confusion matrix.
    """
    if ax is not None:
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
    else:
        plt.figure(figsize=(8,6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)  # Fix: Correct the function name and arguments
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(save_path)

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
    
    # plt.show()

def plot_roc_curve(auc, fpr, tpr, thresholds,save_path="roc_curves"):
    
    plt.figure(figsize=(10, 8))
    plt.subplot(1,2,1)
    plt.plot(thresholds, tpr, label="TPR")
    plt.plot(thresholds,fpr, label="FPR" )

    # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('threshold')
    # plt.ylabel('TPR')
    plt.title(f'AUC = {auc}')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(fpr,tpr)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'AUC = {auc}')
    plt.legend(loc="lower right")
    plt.grid(True)


    task_path = os.path.join(save_path)
    plt.savefig(task_path)

def plot_roc_curves(auc_values, fpr_values, tpr_values, thresholds,save_path="roc_curves"):
    # print(thresholds)
    # print(auc_values)
    for task, auc_val in auc_values.items():
        plt.figure(figsize=(10, 8))
        # print(task)
        # print(thresholds[task])
        plt.subplot(1,2,1)
        plt.plot(thresholds[task], tpr_values[task], label="TPR")
        plt.plot(thresholds[task],fpr_values[task], label="FPR" )

        # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('TPR,FPR')
        plt.title(f'TPR,FPR vs threshold')
        plt.legend(loc="lower right")
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(fpr_values[task],tpr_values[task],label='TPR vs FPR')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'AUROC = {auc_val}')
        plt.suptitle(f'ROC Curves for {task}')
        plt.legend(loc="lower right")
        plt.grid(True)

        task_path = os.path.join(save_path,task+".png")
        plt.savefig(task_path)
    # plt.show()
    # i = np.arange(len(tpr_values)) # index for df
    # roc = pd.DataFrame({'fpr' : pd.Series(fpr_values, index=i),'tpr' : pd.Series(tpr_values, index = i), '1-fpr' : pd.Series(1-fpr_values, index = i), 'tf' : pd.Series(tpr_values - (1-fpr_values), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
    # roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    # # Plot tpr vs 1-fpr
    # fig, ax = pl.subplots()
    # pl.plot(roc['tpr'])
    # pl.plot(roc['1-fpr'], color = 'red')
    # pl.xlabel('1-False Positive Rate')
    # pl.ylabel('True Positive Rate')
    # pl.title('Receiver operating characteristic')
    # ax.set_xticklabels([])
    # plt.savefig("roc_2.png")
    # plt.show()

def plot_precision_recall_curves(auc_pr_values,precision_values,recall_values,thresholds,save_path='pr_curves'):
    for task, auc_pr in auc_pr_values.items():
        plt.figure(figsize=(10, 8))
        plt.subplot(1,2,1)
        plt.plot(thresholds[task], precision_values[task], label="Precision")
        plt.plot(thresholds[task],recall_values[task], label="Recall" )

        # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('Precision,Recall')
        plt.title(f'Precision, Recall vs threshold')
        plt.legend(loc="lower right")
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(recall_values[task],precision_values[task])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision vs  Recall,AUPR = {auc_pr}')
        plt.suptitle(f'Precision - Recall curves for {task} task')
        plt.legend(loc="lower right")
        plt.grid(True)

        task_path = os.path.join(save_path,task+".png")
        plt.savefig(task_path)

