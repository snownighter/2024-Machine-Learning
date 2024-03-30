import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


batch_size = 64
epochs = 5

# normalize = (x - mean) / std
# [0,1] -> [-1,1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

print("Size: ", np.size(trainset[0][0][0]))
print("Split: ", len(trainset) / len(testset))
print("Samples (train): ", len(trainset))
print("Samples (test): ", len(testset))

class Net(nn.Module):
    def __init__(self):
        # network layer
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 16)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        # forward propagation
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.softmax(self.fc2(x), dim=1)
        return x
    
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# train
model.train()

losses = []
accuracies = []
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(
        tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')):
        optimizer.zero_grad()  # 梯度清零
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  # 反向傳播
        optimizer.step()  # 更新權重
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_accuracy = 100 * correct / total
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)

    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
          .format(epoch+1, epochs, epoch_loss, epoch_accuracy))

print('Finished Training')

# test
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {:.2f}%'.format(100 * correct / total))

# 儲存模型
if not os.path.exists('./models'):
    os.makedirs('./models')
torch.save(model.state_dict(), './models/mnist_model.pth')


plt.plot(losses, label='loss')
plt.plot(accuracies, label='accuracy')
plt.xlabel('epoch')
plt.ylabel('acc & loss')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.grid(True)

if not os.path.exists('./img'):
    os.makedirs('./img')
plt.savefig('./img/Training Loss and Accuracy.png')
plt.show()
plt.close()