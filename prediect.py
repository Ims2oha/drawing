import torch
from model import CNN
from dataloader import get_dataset

_, Test_DL = get_dataset(1024)

model = CNN()

model.load_state_dict(torch.load("./model/EPOCH_50_LF_CrossEntropyLoss_optim_Adam_LR_0.001.pth"))

def Test(model,Test_DL):
    model.eval()
    with torch.no_grad():
        rcorrect = 0
        for xb,yb in Test_DL:
            y_hat = model(xb)
            pred = y_hat.argmax(dim=1)
            rcorrect_a = torch.sum(pred == y_batch)
            rcorrect += rcorrect_a
        correct = rcorrect/len(Test_DL.dataset) * 100
    print(f"test_correct: {rcorrect/len(Test_DL.dataset)} ({correct:.1f}%)")
    return correct

Test(model,Test_DL)

