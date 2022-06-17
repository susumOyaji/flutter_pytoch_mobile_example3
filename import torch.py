import torch

# 初期化されていない5☓3行列
x = torch.empty(5,3)

# >> tensor([[2.2037e-35, 0.0000e+00, 3.3631e-44],
#        [0.0000e+00,        nan, 0.0000e+00],
#        [1.1578e+27, 1.1362e+30, 7.1547e+22],
#        [4.5828e+30, 1.2121e+04, 7.1846e+22],
#        [9.2198e-39, 7.0374e+22, 0.0000e+00]])

print(x.shape)
# >> torch.Size([5, 3])

print(x[0])
# >> tensor([2.2037e-35, 0.0000e+00, 3.3631e-44])

print(x[0][0])
# >> tensor(1.6076e-35)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, transforms

# 追加============================
import os
from torchsummary import summary
# ===============================

from datetime import datetime

print(torch.__version__) # 1.5.0

# colabでgoogle driveをマウントしてない場合のパス
root="content/"

# dataの変換方法を定義
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# dataをダウンロード
train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)

# cpuかgpuか
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dataloaderを定義
train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

# Networkを定義
class MLPNet (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 512)
        self.fc2 =nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout1=nn.Dropout2d(0.2)
        self.dropout2=nn.Dropout2d(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return F.relu(self.fc3(x))

net = MLPNet().to(device)

# torchsummaryを使った可視化
summary(net, input_size=(1,1 * 28 * 28))



# 追加部分
dir_name = './assets/models'

if not os.path.exists(dir_name):
    os.mkdir(dir_name)

model_save_path = os.path.join(dir_name, "model_full.pt")

# モデル保存
torch.save(net.state_dict(), model_save_path)

# モデルロード
#model = net()
#model.load_state_dict(torch.load(model_save_path))
#net.load_state_dict(torch.load("assets/models/custom_model.pt"))
model = torch.load("assets/models/custom_model.pt")
print(model)
print(model.state_dict())
#OrderedDict([('fc.weight', tensor([[-0.2375, -0.4745,  0.3979, -0.3354]])), ('fc.bias', tensor([-0.0506]))])
#[1, 2, 3, 4], [1, 2, 2]


# torchsummaryを使った可視化
#summary(model, input_size=(1,1 * 28 * 28))

summary(model, input_size=(3, 416, 416))
