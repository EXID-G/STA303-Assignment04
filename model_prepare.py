import torch
from torch.utils.data import DataLoader,random_split
from torchvision import datasets
import os
from torchvision import transforms
import torch
import torchvision
from torch import nn
import torch.optim as optim
from tqdm import tqdm


# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 4, 3)  
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(4, 8, 3)  
#         self.fc1 = nn.Linear(8 * 6 * 6, 32)
#         self.fc2 = nn.Linear(32, 10)

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(-1, 8 * 6 * 6)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x



def load_resnet18(num_classes):
    # 加载预训练的 ResNet18 模型
    resnet18 = torchvision.models.resnet18(pretrained=True)

    # # 冻结所有层，除了最后一层
    # for param in resnet18.parameters():
    #     param.requires_grad = False

    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
    return resnet18

def load_densenet(num_classes):
    # 加载预训练的 DenseNet 模型
    densenet = torchvision.models.densenet121(pretrained=True)

    # # 冻结所有层，除了最后一层
    # for param in densenet.parameters():
    #     param.requires_grad = False

    # 修改最后一层的全连接层
    in_features = densenet.classifier.in_features
    densenet.classifier = nn.Linear(in_features, num_classes)
    
    return densenet


def training_model(train_dataloader,test_dataloader,model,saving_name,num_epochs = 20):
    # 定义你的数据加载和优化器等
    # train_data_size = len(train_data)
    # test_data_size = len(test_data)
    # print(f"size of training data: {train_data_size}")
    # print(f"size of testing data: {test_data_size}")
    
    if saving_name == None:
        print("saving_name is None, please input a name")
        return
    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 训练
    print(f"----------training for {saving_name}----------")
    for epoch in tqdm(range(num_epochs)):
        # print(f"----------epoch: {epoch+1}----------")
        model.train()
        for image, target in train_dataloader:
            # image = image.to(device)
            # target = target.to(device)
            # model.to(device)
            
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

    # 测试和评估
    model.eval()
    correct = 0
    total = 0
    print(f"----------testing for {saving_name}----------")

    with torch.no_grad():
        for image, target in tqdm(test_dataloader):
            # image = image.to(device)
            # target = target.to(device)
                
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy}')
    
    # 保存模型
    if not os.path.exists("./models"):
        os.mkdir("./models")
    torch.save(model, f"./models/{saving_name}.pth")
    
    model.to('cpu')
    torch.cuda.empty_cache()




    
