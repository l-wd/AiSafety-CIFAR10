import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2 as cv
from model import ResNet18

if __name__ == '__main__':
	# Loading models
	model = ResNet18()
	model.load_state_dict(torch.load('./checkpoint/ResNet18.pth')['model'])
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.to(device)
	# READ IMAGE
	img = cv.imread('./images/9_707.jpg')
	img = img.transpose(2, 0, 1)
	X = torch.Tensor(img.tolist()).unsqueeze(0)
	X = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(X)
	X = X.to(device)


	with torch.no_grad():
		logits = model(X)
		pred = F.softmax(logits, dim=1)
		print(pred[0].argmax(0))
		print(pred)
