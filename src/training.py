import torch


def correct(output, target):
    predicted = output.argmax(1)
    return (predicted == target).type(torch.float).sum().item()


def label_transform_even_odd(labels):
    return labels % 2


def train_epoch(model, loader, optimizer, criterion, device, label_transform=None):
    model.train()
    total_loss, total_correct = 0.0, 0.0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        if label_transform is not None:
            target = label_transform(target)
        result = model(data)
        output = result[0] if isinstance(result, tuple) else result
        loss = criterion(output, target)
        total_loss += loss.item()
        total_correct += correct(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(loader), total_correct / len(loader.dataset)


def evaluate(model, loader, criterion, device, label_transform=None):
    model.eval()
    total_loss, total_correct = 0.0, 0.0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            if label_transform is not None:
                target = label_transform(target)
            result = model(data)
            output = result[0] if isinstance(result, tuple) else result
            loss = criterion(output, target)
            total_loss += loss.item()
            total_correct += correct(output, target)
    return total_loss / len(loader), total_correct / len(loader.dataset)
