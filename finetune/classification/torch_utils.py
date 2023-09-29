import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the device to use for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def numpy_to_data_loader(
    x=None, 
    y=None,
    x_dtype=torch.float32,
    y_dtype=torch.long,
    batch_size=32, 
    shuffle=True,
    num_workers=0,
    pin_memory=False,
):
    """Get a data loader for a dataset."""
    # Convert to tensors
    x = torch.from_numpy(x).type(x_dtype)
    y = torch.from_numpy(y).type(y_dtype)
    dataset = TensorDataset(x, y)
    # Create data loader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader


def model_fit(
    model=None,
    loss_fn=None,
    acc_fn=None,
    optimizer=None,
    train_loader=None,
    epochs=10,
    val_loader=None,
    save_best_only=True,
    early_stopping= 5,
    patience = 5,
    save_every_epoch=False,
    save_path=None,
    device=None,
):
    """Fit a model to a training set."""
    # History
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_loss = np.inf

    # For each training epoch
    for e in range(epochs):
        print("\nEpoch: {}/{}".format(e+1, epochs))

        # Train
        model.train()
        train_losses = []
        train_total_correct = 0
        epoch_bar = tqdm(train_loader)
        for b, batch in enumerate(epoch_bar):
            # get batch data
            x_train, y_train = batch
            x_train, y_train = x_train.to(device), y_train.to(device)
            # forward pass
            optimizer.zero_grad()
            y_pred = model(x_train)

            # backward pass
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()


            # loss
            train_losses.append(loss.item())
            # accuracy
            correct = 0
            if acc_fn is not None:
                correct = acc_fn(y_pred, y_train)
                train_total_correct += correct

            # update progress bar
            epoch_bar.set_description(
                "Epoch: {}, loss {:.5f}, acc: {:.3f}".format(
                    e+1, loss.item(), correct / len(x_train)
                )
            )

            # clear progress bar in the end of each epoch
            if b == len(train_loader) - 1:
                epoch_bar.set_description("")
        
        # update history
        train_loss = np.mean(train_losses)
        train_acc = train_total_correct / len(train_loader.dataset)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc) 

        # Save model (every epoch)
        if save_every_epoch:
            epoch_path = save_path.rsplit(".", 1)[0] + "_" + str(e+1) + ".pt"
            torch.save(model, epoch_path)

        # End without validation data
        if val_loader is None:
            # Print training result
            print('Epoch: {}, loss {}, acc: {}'.format(
                e+1, train_loss, train_acc
            ))

            # Save model (last epoch)
            torch.save(model, save_path)
            continue

        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            val_total_correct = 0
            for idx, batch in enumerate(val_loader):
                # get batch data
                x_val, y_val = batch
                x_val, y_val = x_val.to(device), y_val.to(device)
                # forward pass
                y_pred = model(x_val)
                
                # loss
                loss = loss_fn(y_pred, y_val)
                # print(y_pred, y_val)
                # # print probability
                # print(torch.softmax(y_pred, dim=1), y_val)
                val_losses.append(loss.item())
                # accuracy
                correct = 0
                if acc_fn is not None:
                    correct = acc_fn(y_pred, y_val)
                    val_total_correct += correct

        # update history
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        val_acc = val_total_correct / len(val_loader.dataset)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print training and validation result
        print('Epoch: {}, loss: {}, acc: {}, val_loss {}, val_acc: {}'.format(
            e+1, train_loss, train_acc, val_loss, val_acc
        ))

        # Early stopping
        # if the loss does not decrease for patience consecutive epochs, stop training
        if e > early_stopping:
            if np.all(np.diff(history["val_loss"][e-patience:e]) > 0):
                print("Early stopping")
                break

        # Save model (best model)
        if save_best_only:
            # save if best validation loss gets updated
            if np.isclose(val_loss, best_val_loss):
                torch.save(model, save_path)
        # Save model (last epoch)
        else:
            pass
            # torch.save(model, save_path)

    return history


def model_predict(
    model=None,
    test_loader=None,
    device=None,
):
    """Predict on a test set."""
    # Predict
    model.eval()
    with torch.no_grad():
        y_preds = []
        for idx, batch in enumerate(test_loader):
            # get batch data
            x_test, _ = batch
            x_test = x_test.to(device)
            # forward pass
            y_pred = model(x_test)
            
            y_preds.append(y_pred.cpu().numpy())

    return np.concatenate(y_preds, axis=0)


def model_evaluate(
    model=None,
    loss_fn=None,
    acc_fn=None,
    test_loader=None,
    device=None,
):
    """Evaluate a model on a test set."""
    # Testing
    model.eval()
    with torch.no_grad():
        test_losses = []
        test_total_correct = 0
        for idx, batch in enumerate(test_loader):
            # get batch data
            x_test, y_test = batch
            x_test, y_test = x_test.to(device), y_test.to(device)
            # forward pass
            y_pred = model(x_test)
            # print(nn.Softmax(dim=1)(y_pred))

            # loss
            loss = loss_fn(y_pred, y_test)
            test_losses.append(loss.item())
            # accuracy
            correct = 0
            if acc_fn is not None:
                correct = acc_fn(y_pred, y_test)
                test_total_correct += correct

    # Print testing result
    test_loss = np.mean(test_losses)
    test_acc = test_total_correct / len(test_loader.dataset)
    print('Test loss: {}, acc: {}'.format(test_loss, test_acc))

    return test_loss, test_acc


def classification_acc(type="multi-class"):
    """Return the correct accuracy function for classification."""
    assert type in ["binary", "multi-class"], (
        "Unknown type. Must be one of 'binary' or 'multi-class'."
    )

    def binary_acc(y_pred, y):
        """Compute number of accurate prediction for binary classification."""
        # Round to 0 or 1
        y_pred = torch.round(y_pred)
        # Count number of correct prediction
        correct = (y_pred == y).sum().item()
        return correct
    
    def multi_class_acc(y_pred, y):
        """Compute number of accurate prediction for multi-class classification."""
        # Argmax to get the class
        y_pred = torch.argmax(y_pred, dim=1)
        # print(y_pred)
        # print(y)
        # Count number of correct prediction
        correct = (y_pred == y).sum().item()
        return correct

    if type == "binary":
        return binary_acc
    elif type == "multi-class":
        return multi_class_acc
    else:
        raise ValueError("Unknown classification type: {}".format(type))
