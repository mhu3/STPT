import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models_classification import  SimpleViT, SimpleCNN, SimpleLSTM
from torch_utils import get_device, numpy_to_data_loader
from torch_utils import model_fit, classification_acc, model_evaluate


def main():
    device = get_device()

    # Load pre-processed data
    # train_class the first  4days
    # val_class the last 1day
    x, y = pickle.load(open('dataset/train_class_whole_day_with_status_classification.pkl', 'rb'))

    x_train = x[:480]
    y_train = y[:480]
    x_val = x[480:560]
    y_val = y[480:560]
    x_test = x[560:720]
    y_test = y[560:720]

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    # prepare data
    train_loader = numpy_to_data_loader(x_train, y_train, batch_size=16, shuffle=True)
    val_loader = numpy_to_data_loader(x_val, y_val, batch_size=512, shuffle=False)
    test_loader = numpy_to_data_loader(x_test, y_test, batch_size=512, shuffle=False)

    # Train model
    model = SimpleLSTM(input_size=x_train.shape[2], hidden_size=128, num_layers=9)
    # model = SimpleViT(input_size=x_train.shape[2])
    # model = SimpleCNN(input_size=x_train.shape[2]))

    # Check the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print("Number of available GPUs: ", num_gpus)

    if num_gpus > 1:
        # Wrap the model with nn.DataParallel
        model = nn.DataParallel(model)

    model_tag = "large_LSTM"
    number_tag = 11

    # Define the tag for the pre-trained model (or set it to None if no pre-trained model is available)
    # pretrain_tag = "halfyear_100_20000_30epochs_1"  # Change this value or set it to None
    # pretrain_tag = "halfyear_500_100000_30epochs_1"  # Change this value or set it to None
    # pretrain_tag = "halfyear_1000_400000_30epochs_1"  # Change this value or set it to None
    pretrain_tag = None

    parameters = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {parameters}")
    
    if pretrain_tag is not None:
        pre_trained_model = torch.load(f"pretrainmodels/base_model_ViT_Siamese_with_status_{pretrain_tag}.pt")
        if isinstance(pre_trained_model, nn.DataParallel):
            model_state_dict = pre_trained_model.module.state_dict()
        else:
            model_state_dict = pre_trained_model.state_dict()
        for name in model_state_dict.keys():
            print(name)
        model_dict = model.module.state_dict()
        for name in model_dict.keys():
            print(name)

        parameters = sum(p.numel() for p in model_state_dict.values())
        print(f"Total parameters in pre-trained model: {parameters}")
        
        pre_trained_model_dict = {k: v for k, v in model_state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        # Update the model's state dictionary with the filtered pre-trained model's state dictionary
        model_dict.update(pre_trained_model_dict)

        # Load the modified state dictionary into the model
        model.module.load_state_dict(model_dict)

        borrowed_params = sum(p.numel() for p in pre_trained_model_dict.values())
        print(f"Total parameters borrowed from pre-trained model: {borrowed_params}")

        for name, _ in pre_trained_model_dict.items():
            print(f"Weight borrowed from pre-trained model: {name}")

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    acc_fn = classification_acc("multi-class")
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    EPOCHS = 2000

    history = model_fit(
        model,
        loss_fn,
        acc_fn,
        optimizer,
        train_loader,
        epochs=EPOCHS,
        val_loader=val_loader,
        save_best_only=True,
        early_stopping=10,
        patience=10,
        save_every_epoch=False,
        save_path=f"new_models/{model_tag}_model_5_class_{pretrain_tag}_{number_tag}.pt" if pretrain_tag is not None else f"models/{model_tag}_model_5_class_{number_tag}.pt",
        # save_path="models/{model_tag}_model_5_class.pt",
        device=device,
    )
    model = torch.load(f"new_models/{model_tag}_model_5_class_{pretrain_tag}_{number_tag}.pt" if pretrain_tag is not None else f"models/{model_tag}_model_5_class_{number_tag}.pt")

    evaluate = model_evaluate(
        model,
        loss_fn,
        acc_fn,
        test_loader,
        device=device,
    )
    print(model_tag, pretrain_tag, number_tag)

    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']

    # Save the loss and accuracy if the tag is not None
    if pretrain_tag is not None:
        loss_path = f"new_loss/{model_tag}_5_class_{pretrain_tag}_{number_tag}.pkl"
    else:
        loss_path = f"new_loss/{model_tag}_5_class_{number_tag}.pkl"
    # loss_path = "loss/{model_tag}_5_class.pkl"
    
    pickle.dump([train_loss, val_loss, train_acc, val_acc], open(loss_path, "wb"))

if __name__ == '__main__':
    # set seed
    import numpy as np
    import random
    import torch
    
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # Run main
    main()
