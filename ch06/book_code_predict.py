# Step 1
starter = "This is the starter text"
states = None
import numpy as np

# Step 2
for ch in starter:
    x = np.array([[indexer[ch]]])
    x = index2onehot(x)
    x = torch.Tensor(x)

    pred, states = model(x, states)

# Step 3
counter = 0
while starter[-1] != "." and counter < 50:
    counter += 1
    x = np.array([[indexer[stater[-1]]]])
    x = index2onehot(x)
    x = torch.Tensor(x)

    pred, states = model(x, states)
    pred = F.softmax(pred, dim=1)
    p, top = pred.topk(10)
    p = p.detach().numpy()[0]
    top = top.numpy()[0]

    index = np.random.choice(top, p=p/p.sum())

    # Step 4
    starter += chars[index]
    print(starter)
