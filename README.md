# AiSafety-CIFAR10  
python main.py    
在ResNet18上跑50个epochs准确率：91%，但是太慢了，所以换LetNet5，来验证hyper-parameter的影响  
在LeNet5上跑准确率：  
  epochs：20： 66.5%，batch_size：8  
  epochs：20： 66.5%，batch_size：50  
  epochs：10:  63.1%，batch_size：50  
结论：epochs增加时准确率会上升，但是epochs过多会导致过拟合，且准确率越往后增长越慢，当前epochs还算少。  



