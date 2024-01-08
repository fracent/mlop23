"""This file was created x mlop dtu course 2023,
francesco centomo developed this code with help from
chat gpt4 and mlop material"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Neural network with dropout
class ClassifierWithDropout(nn.Module):
    def __init__(self):
        super(ClassifierWithDropout, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

# Training and validation
def train_and_validate(model, trainloader, testloader):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        accuracy = 0
        with torch.no_grad():
            for images, labels in testloader:
                log_ps = model(images)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        print(f'Epoch {epoch+1}/{epochs} - Accuracy: {accuracy/len(testloader)*100}%')
