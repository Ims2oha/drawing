import torch
from torch import nn,optim
from dataloader import get_dataset
from model import CNN
from train import train

model = CNN()

BATCH_SiZE =  int(input("BATCH_SIZE : "))

train_DL = get_dataset(BATCH_SiZE)

EPOCH = int(input("EPOCH : "))

Loss_fn = nn.CrossEntropyLoss()

LR = float(input("LR : "))

optimizer = optim.Adam(model.parameters,lr=LR)

loss_history = train(EPOCH=EPOCH,Loss_fn=Loss_fn,opimizer=optimizer,model=model,train_DL=train_DL)

print(loss_history)

save_model_path = f"./model/EPOCH_{EPOCH}_LF_{Loss_fn}_optim_{optimizer}_LR_{LR}.pth"

torch.save(model.state_dict(),save_model_path)