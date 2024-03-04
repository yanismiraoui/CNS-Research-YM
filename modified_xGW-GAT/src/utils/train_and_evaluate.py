import logging

import nni
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics

from src.utils.metrics import multiclass_roc_auc_score
from src.utils.save_model import SaveBestModel

# Create logger
logger = logging.getLogger("__name__")
level = logging.INFO
logger.setLevel(level)
ch = logging.StreamHandler()
ch.setLevel(level)
logger.addHandler(ch)

def k_support_norm(x, k):
    epsilon = 1e-10  # Small constant to prevent division by zero
    print("k_support_norm")
    print(x.view(-1))
    x_sorted, _ = torch.sort(torch.square(x.view(-1)), descending=True)
    print(x_sorted)
    cumsum_x_sorted = torch.cumsum(x_sorted, dim=0)
    print(cumsum_x_sorted)
    j = torch.arange(1, x.numel() + 1, dtype=torch.float32, device=x.device)
    print(j)
    # Ensure the denominator is not too small by adding epsilon
    k_support_val = torch.sqrt(cumsum_x_sorted - torch.square(cumsum_x_sorted / (j + epsilon)) + epsilon)
    print(k_support_val)
    k_support_val = torch.max(k_support_val[:k])
    print(k_support_val)

    return k_support_val


def train_eval(
    model, optimizer, scheduler, class_weights, args, train_loader, test_loader=None, sparse_method=None
):
    """
    Train model
    """
    model.train()
    save_best_model = SaveBestModel()  # initialize SaveBestModel class
    criterion = nn.NLLLoss(weight=class_weights)

    train_preds, train_labels, train_aucs, train_accs, train_loss, test_accs, test_aucs, test_losses = [], [], [], [], [], [], [], []
    total_correct = 0
    total_samples = 0
    print("Starting training...")
    if args.sparse_method == "mae":
        print("Training model using sparse method")
    for i in range(args.epochs):
        running_loss = 0  # running loss for logging
        avg_train_losses = []  # average training loss per epoch

        for data in train_loader:
            if sparse_method == "mae":
                data = data.to(args.device)
                optimizer.zero_grad()
                gcn_output, decoded = model(data)
                pred = gcn_output.max(dim=1)[1]  # Get predicted labels
                train_preds.append(pred.detach().cpu().tolist())
                train_labels.append(data.y.detach().cpu().tolist())
                total_correct += int((pred == data.y).sum())
                total_samples += data.y.size(0)  # Increment the total number of samples
                mse_loss = nn.MSELoss()
                reconstruction_loss = mse_loss(decoded, data.x)  # Autoencoder loss
                classification_loss = criterion(gcn_output, data.y)
                if args.model_name == "fc":
                    mask = model.mask
                else:
                    mask = model.sparse_model.mask
                total_loss = classification_loss + 0.01 * torch.linalg.norm(mask) # Frobenius norm
                ##total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(torch.abs(mask))  # L1 regularization 
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(mask ** 2)  # L2 regularization (squared)
                #total_loss = classification_loss + 0.01 * torch.linalg.norm(mask, 1) + 0.01 * torch.linalg.norm(mask, 2) # Elastic Net
                #total_loss = classification_loss + 0.01 * k_support_norm(mask, 5) # K-support norm
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(x**2 * (1 - x**2)) # Forcing to 1 or 0 values
                total_loss.backward()
                optimizer.step()
                running_loss += float(total_loss.item())

            elif sparse_method == "vae":
                data = data.to(args.device)
                optimizer.zero_grad()
                gcn_output, decoded = model(data)
                pred = gcn_output.max(dim=1)[1]  # Get predicted labels
                train_preds.append(pred.detach().cpu().tolist())
                train_labels.append(data.y.detach().cpu().tolist())
                total_correct += int((pred == data.y).sum())
                total_samples += data.y.size(0)  # Increment the total number of samples
                mse_loss = nn.MSELoss()
                reconstruction_loss = mse_loss(decoded, data.x)  # Autoencoder loss
                kl_divergence = -0.5 * torch.sum(1 + model.sparse_model.log_var - model.sparse_model.mu.pow(2) - model.sparse_model.log_var.exp())
                classification_loss = criterion(gcn_output, data.y)
                if args.model_name == "fc":
                    mask = model.mask
                else:
                    mask = model.sparse_model.mask
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.linalg.norm(mask) # Frobenius norm
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(torch.abs(mask))   # L1 regularization 
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(mask ** 2)  # L2 regularization (squared)
                total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * (torch.sum(torch.abs(mask)) + args.weights_elastic * torch.sum(mask ** 2)) # Elastic Net
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * k_support_norm(mask, 5) # K-support norm
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(mask**2 * (1 - mask**2)) # Forcing to 1 or 0 values
                total_loss.backward()
                optimizer.step()
                running_loss += float(total_loss.item())

            elif args.sparse_method == "baseline_mask":
                data = data.to(args.device)
                optimizer.zero_grad()
                gcn_output, masked = model(data)
                pred = gcn_output.max(dim=1)[1]  # Get predicted labels
                train_preds.append(pred.detach().cpu().tolist())
                train_labels.append(data.y.detach().cpu().tolist())
                total_correct += int((pred == data.y).sum())
                total_samples += data.y.size(0)  # Increment the total number of samples
                classification_loss = criterion(gcn_output, data.y)
                if args.model_name == "fc":
                    mask = model.mask
                else:
                    mask = model.sparse_model.mask
                total_loss = classification_loss + 0.01 * torch.linalg.norm(mask) # Frobenius norm
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(torch.abs(mask))  # L1 regularization
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(mask ** 2)  # L2 regularization (squared)
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.linalg.norm(mask, 1) + 0.01 * torch.linalg.norm(mask, 2) # Elastic Net
                #total_loss = classification_loss + reconstruction_loss + 0.01 * k_support_norm(mask, 5) # K-support norm
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(x**2 * (1 - x**2)) # Forcing to 1 or 0 values
                #with torch.autograd.detect_anomaly():
                    #total_loss.backward(retain_graph=True)
                total_loss.backward()
                optimizer.step()
                running_loss += float(total_loss.item())         
            else:
                data = data.to(args.device)
                optimizer.zero_grad()
                out = model(data)
                pred = out.max(dim=0)[1]  # Get predicted labels
                train_preds.append(pred.detach().cpu().tolist())
                train_labels.append(data.y.detach().cpu().tolist())
                total_correct += int((pred == data.y).sum())
                total_samples += data.y.size(0)  # Increment the total number of samples
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())

        avg_train_loss = running_loss / len(
            train_loader.dataset
        )  # Correctly calculate loss per epoch
        avg_train_losses.append(avg_train_loss)

        train_acc, train_auc, _, _, _ = test(model, train_loader, args)

        logging.info(
            f"(Train) | Epoch={i+1:03d}/{args.epochs}, loss={avg_train_loss:.4f}, "
            + f"train_acc={(train_acc * 100):.2f}, "
            + f"train_auc={(train_auc * 100):.2f}"
        )

        if (i + 1) % args.test_interval == 0:
            test_acc, test_auc, test_loss, _, _ = test(model, test_loader, args)
            text = (
                f"(Test) | Epoch {i}), test_acc={(test_acc * 100):.2f}, "
                f"test_auc={(test_auc * 100):.2f}\n"
            )
            logging.info(text)

        if args.enable_nni:
            nni.report_intermediate_result(train_acc)

        if scheduler:
            scheduler.step(avg_train_loss)

        train_accs.append(train_acc)
        train_aucs.append(train_auc)
        train_loss.append(avg_train_loss)

        test_accs.append(test_acc)
        test_aucs.append(test_auc)
        test_losses.append(test_loss)

        save_best_model(avg_train_loss, i, model, optimizer, criterion, args)

    train_accs, train_aucs = np.array(train_accs), np.array(train_aucs)
    return train_accs, train_aucs, train_loss, test_accs, test_aucs, test_losses, model


@torch.no_grad()
def test(model, loader, args, test_loader=None):
    """
    Test model
    """
    model.eval()

    preds = []
    # preds_prob = []
    labels = []
    test_aucs = []
    running_loss = 0
    mse_loss = nn.MSELoss()
    if args.sparse_method == "mae":
        print("Testing model using sparse method")

    for data in loader:
        data = data.to(args.device)

        if args.sparse_method == "mae":
            gcn_output, decoded = model(data)
            pred = gcn_output.max(dim=1)[1]
            reconstruction_loss = mse_loss(decoded, data.x)  # Autoencoder loss
            if args.model_name == "fc":
                    mask = model.mask
            else:
                mask = model.sparse_model.mask
            classification_loss = nn.NLLLoss()(gcn_output, data.y)
            total_loss = classification_loss + reconstruction_loss + 0.01 * torch.linalg.norm(mask) # Frobenius norm
            #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(torch.abs(mask))  # L1 regularization
            #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(mask ** 2)  # L2 regularization (squared)
            #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.linalg.norm(mask, 1) + 0.01 * torch.linalg.norm(mask, 2) # Elastic Net
            #total_loss = classification_loss + reconstruction_loss + 0.01 * k_support_norm(mask, 5) # K-support norm
            #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(x**2 * (1 - x**2)) # Forcing to 1 or 0 values
        elif args.sparse_method == "baseline_mask":
            gcn_output, masked = model(data)
            pred = gcn_output.max(dim=1)[1]
            if args.model_name == "fc":
                    mask = model.mask
            else:
                mask = model.sparse_model.mask
            classification_loss = nn.NLLLoss()(gcn_output, data.y)
            total_loss = classification_loss + 0.01 * torch.linalg.norm(mask) # Frobenius norm
            #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(torch.abs(mask))  # L1 regularization
            #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(mask ** 2)  # L2 regularization (squared)
            #total_loss = classification_loss + 0.01 * torch.linalg.norm(mask, 1) + 0.01 * torch.linalg.norm(mask, 2) # Elastic Net
            #total_loss = classification_loss + 0.01 * k_support_norm(mask, 5) # K-support norm
            #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(x**2 * (1 - x**2)) # Forcing to 1 or 0 values
        elif args.sparse_method == "vae":
            gcn_output, decoded = model(data)
            pred = gcn_output.max(dim=1)[1]
            reconstruction_loss = mse_loss(decoded, data.x)  # Autoencoder loss
            if args.model_name == "fc":
                    mask = model.mask
            else:
                mask = model.sparse_model.mask
            classification_loss = nn.NLLLoss()(gcn_output, data.y)
            kl_divergence = -0.5 * torch.sum(1 + model.sparse_model.log_var - model.sparse_model.mu.pow(2) - model.sparse_model.log_var.exp())
            #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.linalg.norm(mask) # Frobenius norm
            #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(torch.abs(mask))  # L1 regularization 
            #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(mask ** 2)  # L2 regularization (squared)
            total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * (torch.sum(torch.abs(mask)) + args.weights_elastic * torch.sum(mask ** 2)) # Elastic Net
            #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * k_support_norm(mask, 5) # K-support norm
            #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(mask**2 * (1 - mask**2)) # Forcing to 1 or 0 values
        else:
            out = model(data)
            pred = out.max(dim=1)[1]
            total_loss = nn.NLLLoss()(out, data.y)

        preds.append(pred.detach().cpu().numpy().flatten())
        labels.append(data.y.detach().cpu().numpy().flatten())
        running_loss += total_loss.item()
    labels = np.array(labels).ravel()
    preds = np.array(preds).ravel()
    avg_test_loss = running_loss / len(loader.dataset)

    if args.num_classes > 2:
        try:
            # Compute the ROC AUC score.
            t_auc = multiclass_roc_auc_score(labels, preds)
        except ValueError as err:
            # Handle the exception.
            print(f"Warning: {err}")
            t_auc = 0.5
    else:
        t_auc = metrics.roc_auc_score(labels, preds, average="weighted")

    test_aucs.append(t_auc)

    if test_loader is not None:
        _, test_auc, preds, labels = test(model, test_loader, args)
        test_acc = np.mean(np.array(preds) == np.array(labels))

        return test_auc, test_acc
    else:
        t_acc = np.mean(np.array(preds) == np.array(labels))
        return t_acc, t_auc, avg_test_loss, preds, labels
