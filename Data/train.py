"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder
from torchvision import transforms

if __name__=='__main__':
    
    DATA_PROCENT=0.03
    NUM_EPOCHS = 2
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    BLOCK_NUM=[3,4,6,3]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir=r'Data/Words'

    data_transform= transforms.Compose([transforms.Resize(size=(224,224)),
                    transforms.TrivialAugmentWide(num_magnitude_bins=31),
                    transforms.ToTensor()
                    ])
    
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        DatasetDir=data_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE,
        DataProcent=DATA_PROCENT
    )

    model=model_builder.ResNet(model_builder.ResNetblock,3,len(class_names),BLOCK_NUM)
    
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

    engine.plot_loss_curves(results)