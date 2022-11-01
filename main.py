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
from model import GoogLeNet


def train(dataloader, model, loss_func, optimizer, epoch, device):
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)  
		y_hat = model(X)
		loss = loss_func(y_hat, y)
		optimizer.zero_grad()
		loss.backward()       
		optimizer.step()    
		if batch == len(dataloader) -1:
			loss, current = loss.item(), batch * len(X)
			print(f'epoch:{epoch+1}\tloss: {loss:>7f}', end='\t')


# TEST MODEL
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
	print(f'Test Error: Accuracy: {(100*correct):>0.1f}%, Average loss: {test_loss:>8f}\n')

if __name__ == '__main__':
    # DATASET
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # 准备数据集并预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # 导入50000张训练图片
    train_set = datasets.CIFAR10(root='./data', 	 # 数据集存放目录
                                train=True,		 # 表示是数据集中的训练集
                                download=True,  	 # 第一次运行时为True，下载数据集，下载完成后改为False
                                transform=transform_train) # 预处理过程
    
    # 导入10000张测试图片
    test_set = datasets.CIFAR10(root='./data', 
                                train=False,	# 表示是数据集中的测试集
                                download=False,transform=transform_test)
    
    # 加载训练集，实际过程需要分批次（batch）训练                                        
    train_loader = DataLoader(train_set, 	  # 导入的训练集
                            batch_size=50, # 每批训练的样本数
                            shuffle=False,  # 是否打乱训练集
                            num_workers=0)  # 使用线程数，在windows下设置为0
    
    # 加载测试集
    test_loader = DataLoader(test_set, 
                            batch_size=100, # 每批用于验证的样本数
                            shuffle=False, num_workers=0)
    
    # 获取测试集中的图像和标签，用于accuracy计算
    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.next()

    print('dataset epochs: ', len(train_loader)) # batch_size 设置为256，可打印输出40个epoch

    for X, y in train_loader:
        print('X.shape: ', X.shape)
        print('y.shape: ', y.shape)
        break


    # MODEL
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 选择 device 否则默认 CPU
    model = ResNet18().to(device)

    # TRAIN MODEL
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters())
    start_epoch = 0
    epochs = 50
    try:
        if os.path.exists('checkpoint/ResNet18.pth'):
            checkpoint = torch.load(r'checkpoint/ResNet18.pth')
            model.load_state_dict(checkpoint['model'])  #加载模型
            optimizer.load_state_dict(checkpoint['optimizer'])  #加载模型
            start_epoch = checkpoint['epoch']+1 #加载参数
    except BaseException as e:
        print(e)
    print("start_epoch: ",start_epoch)
    print("end_epoch: ",epochs)
    for epoch in range(start_epoch, epochs):
        train(train_loader, model, loss_func, optimizer, epoch, device)
        test(test_loader, model, loss_func, device)
        if (epoch + 1) % 5 == 0:
            state = {'model':model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch':epoch}
            torch.save(state, r'checkpoint/ResNet18.pth')
	# save models
    print('Saved PyTorch LeNet5 State to checkpoint/LeNet5.pth')
 
