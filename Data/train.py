"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils
from torchvision import transforms

def Train(
        data_procent:float,
        num_epochs:int,
        batch_size:int,
        learning_rate:float,
        block_num:list,
        data_dir:str,
        transform:transforms.Compose,
        model_name:str
):
    '''
        Train a pytorch model 

        Args:
            data_procent: Procent of data used for train and test 
            num_epochs: Number of epochs
            batch_size: Number of size of batch
            learning_rate: Learning rate for optimizer
            block_num: List with number of CONV block used in ResNet model
            data_dir: Directory where data is stored
            transform: Transformation used on data
            model_name: file Name to save model

        Example:
            data_procent: 0.99  
            num_epochs: 10 
            batch_size: 32
            learning_rate: 0.01
            block_num: [3,4,6,3]
            data_dir: r'Words'
            transform: transforms.Compose([])
            model_name: ResNet50.pth
    '''
    print(os.getcwd())

    # Porcent wykorzystania całych danych
    DATA_PROCENT=data_procent
    NUM_EPOCHS = num_epochs
    BATCH_SIZE = batch_size
    LEARNING_RATE = learning_rate
    BLOCK_NUM=block_num
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir=data_dir
    print(device)
    
    data_transform=transform

    # data_transform= transforms.Compose([transforms.Resize(size=(224,224)),
    #                 transforms.TrivialAugmentWide(num_magnitude_bins=31),
    #                 transforms.ToTensor()
    #                 ])
    
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        DatasetDir=data_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE,
        DataProcent=DATA_PROCENT
    )

    model=model_builder.ResNet(model_builder.ResNetblock,3,len(class_names),BLOCK_NUM).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                               lr=LEARNING_RATE)

    print(f"len train_dataloader {len(train_dataloader)*32}")
    print(f"len test_dataloader {len(test_dataloader)*32}")


    results=engine.train(model=model,
               train_dataloader=train_dataloader,
               test_dataloader=test_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               epochs=NUM_EPOCHS,
               device=device)
    
    # train_features_batch, train_labels_batch = next(iter(train_dataloader))

    utils.save_model(model=model,
                 target_dir="models",
                 model_name=model_name)
    
    # "ResNet50-Transform224x224-NoShedule-NoDropOut.pth"

    engine.plot_loss_curves(results)
