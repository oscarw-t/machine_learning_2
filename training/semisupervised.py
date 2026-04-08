import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_semisupervised(
    train_features,
    train_labels,
    labeled_indices,
    test_features,
    test_labels,
    device='cuda',
    supervised_epochs=100,
    confidence_threshold=0.90,
):
    epochs = supervised_epochs * 2
    lr = 2.5

    all_x = torch.tensor(train_features, dtype=torch.float32)
    all_y = torch.tensor(train_labels,   dtype=torch.long)

    xtest  = torch.tensor(test_features,  dtype=torch.float32)
    ytest  = torch.tensor(test_labels,    dtype=torch.long)

    X_lab = all_x[labeled_indices]
    y_lab = all_y[labeled_indices]

    head = nn.Linear(train_features.shape[1], 10).to(device)
    fit_head(head, X_lab, y_lab, epochs, lr, device)

    unlabeled_mask = torch.ones(len(train_features), dtype=torch.bool)
    unlabeled_mask[labeled_indices] = False
    X_unlab = all_x[unlabeled_mask]

    head.eval()
    with torch.no_grad():
        probs = torch.softmax(head(X_unlab.to(device)), dim=1).cpu()

    confidence, pseudo_labels = probs.max(dim=1)
    high_conf = confidence >= confidence_threshold

    if high_conf.sum().item() == 0:
        head.eval()
        with torch.no_grad():
            preds = head(xtest.to(device)).argmax(dim=1).cpu()
        return (preds == ytest).float().mean().item() * 100.0

    X_combined = torch.cat([X_lab, X_unlab[high_conf]], dim=0)
    y_combined = torch.cat([y_lab, pseudo_labels[high_conf]], dim=0)

    head2 = nn.Linear(train_features.shape[1], 10).to(device)
    fit_head(head2, X_combined, y_combined, epochs, lr, device)

    head2.eval()
    with torch.no_grad():
        preds = head2(xtest.to(device)).argmax(dim=1).cpu()

    return (preds == ytest).float().mean().item() * 100.0


def fit_head(head, X, y, epochs, lr, device):
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=min(64, len(y)),
        shuffle=True,
    )
    optimizer = optim.SGD(head.parameters(), lr=lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    head.train()
    
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            loss = criterion(head(x_batch.to(device)), y_batch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
