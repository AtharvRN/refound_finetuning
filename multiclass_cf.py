from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch
import itertools
# import torch.nn as nn
# import torch.optim as optim
from data import create_data_loaders
import models_vit
# from util.pos_embed import interpolate_pos_embed
# from timm.models.layers import trunc_normal_
import numpy as np
import logging
from torchvision import models
import matplotlib.pyplot as plt
def calculate_roc_auc(model, data_loader, device='cuda', tasks=["foveal_scan", "healthy", "srf", "irf", "drusen", "hdots", "hfoci", "ped"]):
    model.eval()
    
    fpr_values = {}
    tpr_values = {}
    auc_values = {}
    with torch.no_grad():
        for i,task in enumerate(tasks):
            all_labels = []
            all_predictions = []

            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = {key: value.to(device) for key, value in labels.items()}  # Move all labels to device
                task_labels = labels[task].cpu().numpy()
                # print(task_labels.shape)
                all_labels.extend(task_labels)

                outputs = model(inputs)
                outputs = torch.sigmoid(outputs).cpu().numpy()
                # print(outputs.shape)
                all_predictions.extend(outputs[:,i])
            
            fpr, tpr, _ = roc_curve(all_labels, all_predictions)
            roc_auc = auc(fpr, tpr)
            
            fpr_values[task] = fpr
            tpr_values[task] = tpr
            auc_values[task] = roc_auc

    return auc_values,fpr_values,tpr_values

def plot_roc_curves(auc_values, fpr_values, tpr_values, save_path="roc_curves.png"):
    plt.figure(figsize=(10, 8))
    for task, auc_val in auc_values.items():
        plt.plot(fpr_values[task], tpr_values[task], label=f"{task} (AUC = {auc_val:.2f})")

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Tasks')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

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

def get_confusion_matrices(model, test_loader, device='cuda', tasks=["foveal_scan", "healthy", "srf", "irf", "drusen", "hdots", "hfoci", "ped"], threshold=0.5):
    model.eval()
    
    num_tasks = len(tasks)
    confusion_matrices = {}
    
    with torch.no_grad():
        for task in tasks:
            all_labels = []
            all_predictions = []

            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = {key: value.to(device) for key, value in labels.items()}  # Move all labels to device
                task_labels = labels[task].cpu().numpy()
                all_labels.extend(task_labels)

                outputs = model(inputs)
                task_predictions = (torch.sigmoid(outputs)[:, tasks.index(task)] > threshold).cpu().numpy()
                all_predictions.extend(task_predictions)

            confusion_matrices[task] = confusion_matrix(all_labels, all_predictions)

    return confusion_matrices


# Main function
def main():
    # Define hyperparameters
    batch_size = 32
    logging.basicConfig(filename='training_logs_multiclass_models_retfound_confusion_matrix.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

     # Define paths and filenames
    train_csv_file = "/home/tejadhith/Project/OCT/Dataset/28-09-2023_download.csv"
    test_csv_file = "/home/tejadhith/Project/OCT/Dataset/test_data.csv"
    train_root_dir = "/home/tejadhith/Project/OCT/Dataset/segregated_28-sep-2023_kath"
    test_root_dir =  "/home/tejadhith/Project/OCT/Dataset/stavan_images_March16"
    model_save_root_dir = "/home/tejadhith/Project/OCT/multiclass_models_retfound_448"

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_csv_file, train_root_dir, test_csv_file,test_root_dir, batch_size=batch_size,image_size=224)
    # print("Length of Train Dataset : ",len(train_loader.dataset))
    # print("Length of Val Dataset : ",len(val_loader.dataset))
    print("Length of Test Dataset : ",len(test_loader.dataset))

    # call the model
    model = models_vit.__dict__['vit_multi_head_classifiers'](
    img_size = 224,
    num_outputs = 8,
    drop_path_rate=0.2,
    global_pool=True,
)
    
    # load RETFound weights
    checkpoint = torch.load('RETFound_oct_weights.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']
    # print(checkpoint_model['pos_embed'].shape)
    # print(checkpoint_model['head.bias'])
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    # print(model.pos_embed.shape)
    # interpolate_pos_embed(model, checkpoint_model)

    # # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    
    # load RETFound weights
    # checkpoint = torch.load('/home/tejadhith/Project/OCT/multiclass_models_resnet50/best_model.pth', map_location='cpu')
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

    PATH = "/home/tejadhith/Project/OCT/multiclass_models_retfound_448/best_model.pth"
    model.load_state_dict(torch.load(PATH,map_location="cpu"))

    # Define loss function and optimizer
    # model = OCT_Classifier(num_classes=8)

    tasks = ["foveal_scan","healthy","srf","irf","drusen","ped","hdots","hfoci"]
        
    test_accuracy_final = calculate_accuracy(model, test_loader, device='cuda',tasks=tasks)
    for i, task in enumerate(tasks):
        print(f" Final Test Accuracy ({task}): {test_accuracy_final[i]:.4f}")

    cf_matrices = get_confusion_matrices(model, test_loader, 'cuda', tasks, 0.5)
    
    # Save confusion matrices as images
    save_path = 'confusion_matrices_0_5_latest.png'

    plot_confusion_matrices(cf_matrices, tasks,save_path)

    auc_values,fpr_values,tpr_values = calculate_roc_auc(model, test_loader, device='cuda')
    print(fpr_values)
    print(tpr_values)
    plot_roc_curves(auc_values,fpr_values,tpr_values ,save_path="ROCcurves.png")
    # for task, cm in cf_matrices.items():
    #     save_path = f'confusion_matrix_{task}.png'
    #     plot_confusion_matrix(cm, classes=["Negative", "Positive"], title=f'Confusion matrix for {task}', save_path=save_path)

    # for key in cf_matrices.keys():
    #     print("Task : ",key)
    #     print(cf_matrices[key])
    # print(cf_matrices)


if __name__ == "__main__":
    main()
# Call this function after your main training loop to get confusion matrices
