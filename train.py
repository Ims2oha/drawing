import torch


def train(model,train_DL,opimizer,Loss_fn,EPOCH):
    model.train()
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda",enabled=use_amp)
    torch.backends.cudnn.benchmark = True
    loss_history = []
    NoT = len(train_DL.dataset)
    for ep in range(EPOCH):
        rloss = 0
        for xb,yb in train_DL:
            xb,yb = xb.to(device,non_blocking=True),yb.to(device,non_blocking=True)
            opimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type,enabled=use_amp):
                y_hat = model(xb)
                loss = Loss_fn(y_hat,yb)
            scaler.scale(loss).backward()
            scaler.step(opimizer)
            scaler.update()
            loss_b = loss.detach() * xb.shape[0]
            rloss += loss_b
        loss_e = (rloss/NoT).item()
        loss_history += [loss_e]
        print(f"EPOCH: {ep+1} loss : {loss_e}")
        print("-"*20)
    return loss_history
