import torch
import torch.nn as nn
import torch.nn.functional as F
import theano




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(4,1)

    def forward(self, X):
        #shape -> [1,2,2], shape[0] is batch_size
        X = X.view(X.shape[0], -1)
        return torch.sigmoid(self.fc(X))


num_classes = 10
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ネットワークのモジュール化
class Net2(nn.Module):
    def __init__(self, input):
        super(Net2, self).__init__()
        
        # ネットワークを定義
        self.linear1 = nn.Linear(4, 10)
        self.linear2 = nn.Linear(10, 8)
        self.linear3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    # 順伝搬を定義
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x  




if __name__ == "__main__":
    model = Net2("1")
    print(model)
    model.eval()
    print(model)
    example = torch.rand(1,2,2) #tensor of size input_shape
    print(example)
    traced_script_module = torch.jit.trace(model, example)
    #traced_script_module.save("example/assets/models/custom_model.pt")
    traced_script_module.save("testmodels/custom_model.pt")


    from torch.autograd import Variable
    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import torch.optim as optim

    iris = datasets.load_iris()
    y = np.zeros((len(iris.target), 1 + iris.target.max()), dtype=int)
    y[np.arange(len(iris.target)), iris.target] = 1
    X_train, X_test, y_train, y_test = train_test_split(iris.data, y, test_size=0.25)
    x = Variable(torch.from_numpy(X_train).float(), requires_grad=True)
    y = Variable(torch.from_numpy(y_train).float())
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for i in range(3000):
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # test
    outputs = net(Variable(torch.from_numpy(X_test).float()))
    _, predicted = torch.max(outputs.data, 1)
    y_predicted = predicted.numpy()
    y_true = np.argmax(y_test, axis=1)
    accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
    print('accuracy: {0}%'.format(accuracy))


    # utility function to predict for an unknown data
    def predict(X):
        X = Variable(torch.from_numpy(np.array(X)).float())
        outputs = net(X)
        return np.argmax(outputs.data.numpy())
      