import pickle
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from sklearn.model_selection import train_test_split

from models_ss import binaryLSTM, binaryCNN, binaryViT
from torch_utils import get_device, numpy_to_data_loader
from torch_utils import model_fit, classification_acc, model_evaluate

def main():
    device = get_device()

    # Load pre-processed data
    # train_class the first  4days
    # val_class the last 1day
    x_train, y_train = pickle.load(open('data_filtered/train_seek_and_serve_pad_07_2000_2020.pkl', 'rb'))
    # x_val, y_val = pickle.load(open('data_filtered/val_seek_and_serve_pad_07_2000_2020.pkl', 'rb'))
    print(x_train.shape, y_train.shape)

    # # #train,val,test split
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.6, random_state=42, stratify=y_train)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    y_train = y_train[:, np.newaxis]
    y_val = y_val[:, np.newaxis]
    y_test = y_test[:, np.newaxis]


    # normalize the data
    x_train[:, :, 0] = x_train[:, :, 0] / 92
    x_train[:, :, 1] = x_train[:, :, 1] / 49
    x_train[:, :, 2] = x_train[:, :, 2] / 288

    x_val[:, :, 0] = x_val[:, :, 0] / 92
    x_val[:, :, 1] = x_val[:, :, 1] / 49
    x_val[:, :, 2] = x_val[:, :, 2] / 288

    x_test[:, :, 0] = x_test[:, :, 0] / 92
    x_test[:, :, 1] = x_test[:, :, 1] / 49
    x_test[:, :, 2] = x_test[:, :, 2] / 288
    # print how many 0 and 1 in train, val, test
    print(np.sum(y_train == 0), np.sum(y_train == 1))
    print(np.sum(y_val == 0), np.sum(y_val == 1))
    print(np.sum(y_test == 0), np.sum(y_test == 1))

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    train_loader = numpy_to_data_loader(x_train, y_train, y_dtype = torch.float32, batch_size=16, shuffle=True)
    val_loader = numpy_to_data_loader(x_val, y_val, y_dtype = torch.float32, batch_size=512, shuffle=False)
    test_loader = numpy_to_data_loader(x_test, y_test, y_dtype = torch.float32, batch_size=512, shuffle=False)

    # Train model
    # model = binaryLSTM(input_size=x_train.shape[2], hidden_size=128, num_layers=1)
    # model = binaryCNN(input_size=x_train.shape[2])
    model = binaryViT(input_size=x_train.shape[2])

    model_tag = 'ViT_Base'

    # pretrain_tag = 'halfyear_100_20000_30epochs_1'
    # pretrain_tag = 'halfyear_500_100000_30epochs_1'
    # pretrain_tag = 'halfyear_1000_400000_30epochs_1'
    pretrain_tag = None

    parameters = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {parameters}")

    if pretrain_tag is not None:
        pre_trained_model = torch.load(f"pretrainmodels/large_model_ViT_Siamese_with_status_{pretrain_tag}.pt")
        if isinstance(pre_trained_model, nn.DataParallel):
            model_state_dict = pre_trained_model.module.state_dict()
        else:
            model_state_dict = pre_trained_model.state_dict()
        
        parameters = sum(p.numel() for p in model_state_dict.values())
        print(f"Total parameters in pre-trained model: {parameters}")
        
        model_dict = model.state_dict()
        
        pre_trained_model_dict = {
            k: v for k, v in model_state_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
            and not k.startswith('model.fc.0.bias')
        }

        # #don't include the positional embedding
     
        pre_trained_pos_embedding = model_dict['model.pos_embedding']
        pre_trained_pos_embedding = torch.zeros_like(pre_trained_pos_embedding)
        print(pre_trained_pos_embedding.shape)
        pre_trained_model_dict['model.pos_embedding'] = pre_trained_pos_embedding

        # Modify input_layer.weight shape
        input_layer_weight_shape = model_dict['model.input_layer.weight'].shape
        pre_trained_input_layer_weight = model_state_dict['model.input_layer.weight']
        print(input_layer_weight_shape, pre_trained_input_layer_weight.shape)
        if input_layer_weight_shape != pre_trained_input_layer_weight.shape:
            pre_trained_input_layer_weight = pre_trained_input_layer_weight[:, :input_layer_weight_shape[1]]
        print(pre_trained_input_layer_weight.shape)
        pre_trained_model_dict['model.input_layer.weight'] = pre_trained_input_layer_weight

        
        model_dict.update(pre_trained_model_dict)
        model.load_state_dict(model_dict)

        borrowed_params = sum(p.numel() for p in pre_trained_model_dict.values())
        print(f"Total parameters borrowed from pre-trained model: {borrowed_params}")

        for name, _ in pre_trained_model_dict.items():
            print(f"Weight borrowed from pre-trained model: {name}")


    model.to(device)
    loss_fn = nn.BCELoss()
    acc_fn = classification_acc("binary")
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    EPOCHS = 400
    
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
        patience = 10,
        save_every_epoch=False,
        save_path=f'new_models/{model_tag}_model_seek_serve_pretrain_{pretrain_tag}.pt' if pretrain_tag is not None else f'{model_tag}_model_seek_serve.pt',
        # save_path=f'{model_tag}_model_seek_serve.pt',
        device=device,
    )

    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    
    model = torch.load(f'new_models/{model_tag}_model_seek_serve_pretrain_{pretrain_tag}.pt' if pretrain_tag is not None else f'{model_tag}_model_seek_serve.pt')
    # Evaluate model
    evaluate = model_evaluate(
        model,
        loss_fn,
        acc_fn,
        test_loader,
        device=device,
    )
    
    # print model tag , pretrain_tag, and test loss
    print(model_tag, pretrain_tag)

    # Save the loss and accuracy if the tag is not None
    if pretrain_tag is not None:
        loss_path = f"new_loss/{model_tag}_seek_serve_{pretrain_tag}_07_2000_2020.pkl"
    else:
        loss_path = f"new_loss/{model_tag}_seek_serve_07_2000_2020.pkl"
    # loss_path = "loss/{model_tag}_5_class.pkl"
    
    pickle.dump([train_loss, val_loss, train_acc, val_acc], open(loss_path, "wb"))

    # pickle.dump([train_loss, val_loss, train_acc, val_acc], open('loss/{model_tag}_seek_serve_{pretrain_tag}.pkl', 'wb'))

    EPOCHS = len(train_loss)

    # Draw figure to plot the loss and accuracy
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCHS), train_loss, label='train')
    plt.plot(range(EPOCHS), val_loss, label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCHS), train_acc, label='train')
    plt.plot(range(EPOCHS), val_acc, label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    if pretrain_tag is not None:
        plt.savefig(f'new_figure/{model_tag}_seek_serve_{pretrain_tag}_07_2000_2020.png')
    else:
        plt.savefig(f'new_figure/{model_tag}_seek_serve_07_2000_2020.png')

if __name__ == '__main__':
    # set seed
    import numpy as np
    import random
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Run main
    main()


    
