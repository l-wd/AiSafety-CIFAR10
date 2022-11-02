from distutils.log import error
from logging import exception
import os
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from model import LeNet5
from model import ResNet18


def train(dataloader, model, loss_func, optimizer, epoch, device):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        loss = loss_func(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch == len(dataloader) - 1:
            loss, current = loss.item(), batch * len(X)
            print(f'epoch:{epoch+1}\tloss: {loss:>7f}', end='\t')


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f'Test Error: Accuracy: {(100*correct):>0.1f}%, Average loss: {test_loss:>8f}\n')


if __name__ == '__main__':

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(root='./data',
                                 train=True,
                                 download=True,
                                 transform=transform_train)

    test_set = datasets.CIFAR10(root='./data',
                                train=False,
                                download=False, transform=transform_test)

    train_loader = DataLoader(train_set,
                              batch_size=64,
                              shuffle=False,
                              num_workers=0)

    test_loader = DataLoader(test_set,
                             batch_size=100,
                             shuffle=False, num_workers=0)

    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.next()

    print('dataset epochs: ', len(train_loader))

    for X, y in train_loader:
        print('X.shape: ', X.shape)
        print('y.shape: ', y.shape)
        break
    model_name = 'LeNet5'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_name == 'LeNet5':
        model = LeNet5().to(device)
    elif model_name == 'ResNet18':
        model = ResNet18().to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters())
    start_epoch = 0
    epochs = 20
    try:
        if os.path.exists(f'checkpoint/{model_name}.pth'):
            checkpoint = torch.load(f'checkpoint/{model_name}.pth')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']+1
    except BaseException as e:
        print(e)
    print("start_epoch: ", start_epoch)
    print("end_epoch: ", epochs)
    for epoch in range(start_epoch, epochs):
        train(train_loader, model, loss_func, optimizer, epoch, device)
        test(test_loader, model, loss_func, device)
        if (epoch + 1) % 5 == 0:
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch}
            torch.save(state, f'checkpoint/{model_name}.pth')

    print(f'Saved PyTorch {model_name} State to checkpoint/{model_name}.pth')
