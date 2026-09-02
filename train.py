def train(model,train_DL,opimizer,Loss_fn,EPOCH):
    model.train()
    loss_history = []
    NoT = len(train_DL.dataset)
    for ep in range(EPOCH):
        rloss = 0
        for xb,yb in train_DL:
            y_hat = model(xb)
            loss = Loss_fn(y_hat,yb)
            opimizer.zero_grad()
            loss.backward()
            opimizer.step()
            loss_b = loss.item() * xb.shape[0]
            rloss += loss_b
            loss_e = rloss/NoT
        loss_history += [loss_e]
        print(f"EPOCH: {ep+1} loss : {loss_e}")
        print("-"*20)
    return loss_history