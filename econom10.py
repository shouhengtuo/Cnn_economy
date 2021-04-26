# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 08:57:25 2021

@author: Shouheng Tuo
"""
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
from tensorboardX import SummaryWriter
import numpy as np
from thop import profile
from graphviz import Digraph
from torch.utils import data # 获取迭代数据
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)

from scipy import stats

def make_dot(var, params=None):
    """
    画出 PyTorch 自动梯度图 autograd graph 的 Graphviz 表示.
    蓝色节点表示有梯度计算的变量Variables;
    橙色节点表示用于 torch.autograd.Function 中的 backward 的张量 Tensors.

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled', shape='box', align='left',
                              fontsize='12', ranksep='0.1', height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            elif var in output_nodes:
                dot.node(str(id(var)), str(type(var).__name__), fillcolor='darkolivegreen1')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    # 多输出场景 multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)
    return dot

## 载入数据
#归一化之前
#min:16.88	31.92	28.44
#max-min:384238.42	553944.88	77725.66
Min = np.array([0,0,0])
MaxMin = np.array([44226.400000,62540.780000,5556.600000])
Region = np.array(['安徽','北京','福建','江西','辽宁','内蒙古','宁夏','青海',
                   '山东','山西','陕西','上海','四川','天津','西藏','新疆','云南',
                   '浙江','重庆','海南','河北','河南','黑龙江','湖北','湖南',
                   '吉林','江苏','甘肃','广东','广西','贵州'])

f = pd.read_csv("economic7.csv",encoding='utf-8')
fdata = np.array(f)
fdata2 = np.zeros(shape=(len(fdata),5))
for i in range(len(fdata)):
    a = np.array(fdata[i])[0].split("\t")
    fdata2[i,:] = np.array(a[2:7],dtype="float")
    

# fig = plt.figure()
# res = stats.probplot(fdata2[:,3], plot=plt) #默认检测是正态分布
# plt.ylabel('有序值 Order Values')
# plt.xlabel('理论分位数 Theoretical quantiles')
# plt.title('概率图')
# plt.show()


Xtrain = np.empty((589,1,5,5))
Ytrain = np.empty((589,3))
x0 = np.empty((((31-14) + 14*5)*19,1,5,5))
x1 = np.empty((((31-14) + 14*5)*19,1,5,5))
y0 = np.empty((((31-14) + 14*5)*19,3))
y1 = np.empty((((31-14) + 14*5)*19,3))

Xtab = [1,2,4,5,6,7,13,14,15,16,19,22,25,27]

T = 24;
c = 0
c2 = 0
for i in range(31):
    for t in range(T-5):
        sb = i*T + t
        se = sb + 5
        print(sb)
        print(se)
        x = np.array(fdata2[sb:se,:5]);
        y = np.array(fdata2[se,2:5]);
        Xtrain[c] = x;
        Ytrain[c] = y;  
        if i not in Xtab:
            x0[c2] = ((np.random.rand(1,5,5)-0.5) * 0.0001 + 1) * Xtrain[c]
            x1[c2] = ((np.random.rand(1,5,5)-0.5) * 0.0001 +1) * Xtrain[c]
            y0[c2] = Ytrain[c] * (1 + (np.random.rand(1,3)-0.5) * 0.0001)
            y1[c2] = Ytrain[c] * (1 + (np.random.rand(1,3)-0.5) * 0.0001)         
            c2 = c2 + 1
        else:
            for k in range(5):
                x0[c2] = ((np.random.rand(1,5,5)-0.5) * 0.0001 + 1) * Xtrain[c]
                x1[c2] = ((np.random.rand(1,5,5)-0.5) * 0.0001 +1) * Xtrain[c]
                y0[c2] = Ytrain[c] * (1 + (np.random.rand(1,3)-0.5) * 0.0001)
                y1[c2] = Ytrain[c] * (1 + (np.random.rand(1,3)-0.5) * 0.0001)
                c2 = c2 + 1
        c = c + 1

Xtrain0 = np.vstack([Xtrain,x0,x1])
Ytrain0 = np.vstack([Ytrain,y0,y1])
        
train_dataset0 = data.TensorDataset(torch.Tensor(Xtrain0),torch.Tensor(Ytrain0))    
train_loader0 = data.DataLoader(train_dataset0,batch_size=100,shuffle=True)

train_dataset = data.TensorDataset(torch.Tensor(Xtrain),torch.Tensor(Ytrain))          
train_loader = data.DataLoader(train_dataset,batch_size=40,shuffle=True)
test_loader = data.DataLoader(train_dataset,batch_size=19,shuffle=False)  
## 自定义损失函数

class My_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x,y):
        #return 100*(torch.max(torch.pow((x-y),2)) - torch.min(torch.pow((x-y),2))) #+ torch.mean(torch.pow(10*(x-y),2))
        # return torch.sum(torch.pow((x-y),2))
        #return torch.mean(torch.sum(torch.pow(10000*(x-y),2)))
        # return torch.nn.SmoothL1Loss()
        return torch.sum(torch.log(torch.cosh(20*(x-y))))
## 定义CNN network
class CNNnetwork(torch.nn.Module):
    def __init__(self):
        super(CNNnetwork,self).__init__()
        self.conv1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, 
                            out_channels=64, 
                            kernel_size=(3,1),
                            stride = 1,
                            padding = (2,1)),
            # torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3,1))
            )
        self.conv1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, 
                            out_channels=32, 
                            kernel_size=(1,3),
                            stride = 1,
                            padding = (1,2)),
            # torch.nn.BatchNorm2d(32),
            # torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(1,2))            
            )
     
        
        self.conv1_3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, 
                            out_channels=16, 
                            kernel_size=(2,2),
                            stride = 1,
                            padding = (0,0)),
            # torch.nn.BatchNorm2d(16),
            # torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,2))
            )
        self.conv0_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size =(3,1),
                            padding =(1,0)
                            ),
            # torch.nn.BatchNorm2d(16),
            # torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3,1))            
            )
      
        
        inSize = 16
        hideSize = 100
        outSize = 3
        
        self.mlp0 = torch.nn.Sequential(
            torch.nn.Linear(inSize,hideSize),
            # torch.nn.ReLU()
            )
        self.mlp00 = torch.nn.Sequential(
            torch.nn.Linear(hideSize,outSize),
            # torch.nn.ReLU()
            # torch.nn.Sigmoid()
            )
        
        
        self.mlp1_1 = torch.nn.Sequential(
            torch.nn.Linear(inSize+outSize, hideSize ),
            # torch.nn.Dropout(0.5),
            # torch.nn.Sigmoid()
            )
        
        self.mlp1_2 = torch.nn.Sequential(
            torch.nn.Linear(hideSize , outSize),
            # torch.nn.BatchNorm1d(outSize),
            # torch.nn.Dropout(0.5),
            # torch.nn.Sigmoid()
            )
        # self.dropout = torch.nn.Dropout(0.5)
        
        
        
    def forward(self,x):   
        xsize = x.shape
        x0 = x[:,:,1:5,1:5]
        x00 = x[:,:,:,0]
        x00 = x00.view(xsize[0],1,5,1)
        
        x0 = self.conv1_1(x0)
        
        x0 = self.conv1_2(x0)
        
        x0 = self.conv1_3(x0)
        
        x0 = torch.squeeze(x0)  
        
        
        x00 = self.conv0_1(x00)
     
        x00 = torch.squeeze(x00)
        
        x00 = self.mlp0(x00)
        x00 = self.mlp00(x00)
        
        
        
        x0 = torch.cat((x0,x00),1)    
       
        
        x1 = self.mlp1_1(x0)               
        x1 = self.mlp1_2(x1)
      
                    
        return x1
    

model = CNNnetwork()
use_GPU = torch.cuda.is_available()
# dummy_input = Variable(torch.Tensor(2,1,5,5), requires_grad = True)  

# dummy_input = Variable(torch.rand(2,1,5,5)) #假设输入2张1*8*5的图片:tensorboard --logdir runs
# with SummaryWriter(comment='CNNnetwork') as w:
#     w.add_graph(model, (dummy_input, ))


# flops, params = profile(model,inputs = (dummy_input))
# print(model)
# vis_graph = make_dot(model(dummy_input), params=dict(model.named_parameters()))
# vis_graph.view()

# loss_func = torch.nn.MSELoss(reduction='sum')#L1Loss(reduction='sum') 
# loss_func = My_loss()
loss_func = torch.nn.SmoothL1Loss()
loss_func = torch.nn.MSELoss()
loss_func = My_loss()

# loss_func = torch.nn.MSELoss(reduction='sum')
if(use_GPU):
    model = model.cuda()
    # loss_func = My_loss()
    loss_func = loss_func.cuda()
    # loss_func1 = loss_func1.cuda()

opt2 = torch.optim.AdamW(model.parameters(),lr=0.0001,betas=(0.95,0.999) , eps=1e-8)
# scheduler_1 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda=lambda epoch: 1/(epoch+1))
# opt2 = torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.8, amsgrad=True)
# opt2 = torch.optim.RMSprop(model.parameters(),lr=0.0001)
# opt2 = torch.optim.SGD(model.parameters(), lr=0.00001,momentum=0.9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
# opt2 = torch.optim.AdamW(model.parameters(),lr=0.001)

# torch.optim.lr_scheduler.StepLR(opt2, 10, gamma=0.1, last_epoch=-1)
loss_count = []

# 
for epoch in range(3000):
   model.train() 
   for i,(x,y) in enumerate(train_loader0):
        batch_x = Variable(x)
        batch_y = Variable(y)
        if(use_GPU):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
               
        y1= model(batch_x)
        loss = loss_func(y1,batch_y)

        
        
        opt2.zero_grad()
       
        loss.backward()
                  
        opt2.step()
        # scheduler_1.step()
        if i%20 == 0:
            loss_count.append(loss)
            
            # loss4_count.append(loss4)
            print('epoch{}:{}\t'.format(epoch,i), loss.item())
            torch.save(model,r'.\log_CNN')
        # if i % 100 == 0:
        #     for a,b in test_loader:
        #         test_x = Variable(a).cuda()
        #         test_y = Variable(b).cuda()
        #         outY = model(test_x)
        #         # print('test_out:\t',torch.max(out,1)[1])
        #         # print('test_y:\t',test_y)
        #         accuracy = torch.sum(torch.pow(outY - test_y,2))
        #         print('----accuracy:\t',accuracy.mean())
        #         break
plt.figure('PyTorch_CNN_Loss')
plt.semilogy(loss_count,label='Loss')
plt.legend()
plt.show()

# plt.figure('PyTorch_CNN_Loss1')
# plt.plot(loss1_count,label='Loss1')
# plt.legend()
# plt.show()

# plt.figure('PyTorch_CNN_Loss2')
# plt.plot(loss2_count,label='Loss2')
# plt.legend()
# plt.show()

# plt.figure('PyTorch_CNN_Loss3')
# plt.plot(loss3_count,label='Loss3')
# plt.legend()
# plt.show()

# plt.figure('PyTorch_CNN_Loss4')
# plt.plot(loss4_count,label='Loss4')
# plt.legend()
# plt.show()

# Y1 = []
# for i,(x,y) in enumerate(train_loader):
#         batch_x = Variable(x)
#         batch_y = Variable(y)
#         if(use_GPU):
#             batch_x = batch_x.cuda()
#             batch_y = batch_y.cuda()
#         [y1,y2,y3] = model(batch_x)
#         Y1.append(y1)

plt.figure('predict values1')

plt.plot(y1.detach().cpu().numpy(),'bo-')
plt.plot(batch_y.detach().cpu().numpy(),'r*-', label="loss1")
plt.show()

plt.figure()
plt.plot(y1[:,0].detach().cpu().numpy(),'bo-')
plt.plot(batch_y[:,0].detach().cpu().numpy(),'r*-', label="loss1")
plt.show()

# 展示每个区域的预测情况
matplotlib.rcParams['font.sans-serif']=['SimHei']  #使用指定的汉字字体类型（此处为黑体）
xtime = torch.linspace(2002,2020,19)
MSEDATA0 = np.zeros((31,3))
MSEDATA = np.zeros((31,3))
R2 = np.zeros((31,3))
MAE = np.zeros((31,3))
RMSE0 = np.zeros((31,3))
R20 = np.zeros((31,3))
MAE0 = np.zeros((31,3))
RMSLE = np.zeros((31,3))  #(Root Mean Squared Logarithmic Error)
for i,(x,y) in enumerate(test_loader):
    batch_x = Variable(x)
    batch_y = Variable(y)
    if(use_GPU):
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
    y = model(batch_x)
    
    # 还原数据 max_min * x^5 + min; x^5是原数据 是被 x^0.2了
    MSEDATA0[i,1] = torch.mean(torch.pow(torch.pow(y[:,0],5) - torch.pow(batch_y[:,0],5),2))
    MSEDATA0[i,2] = torch.mean(torch.pow(torch.pow(y[:,1],5) - torch.pow(batch_y[:,1],5),2))
    MSEDATA0[i,0] = torch.mean(torch.pow(torch.pow(y[:,2],5) - torch.pow(batch_y[:,2],5),2))
    RMSE0[i,1] =  torch.sqrt(torch.mean(torch.pow(torch.pow(y[:,0],5) - torch.pow(batch_y[:,0],5),2)))
    
    MAE0[i,1] = torch.mean(torch.abs(torch.pow(y[:,0],5) - torch.pow(batch_y[:,0],5)))
    MAE0[i,2] = torch.mean(torch.abs(torch.pow(y[:,1],5) - torch.pow(batch_y[:,1],5)))
    MAE0[i,0] = torch.mean(torch.abs(torch.pow(y[:,2],5) - torch.pow(batch_y[:,2],5)))
    
    a = torch.sum(torch.pow(torch.pow(y[:,0],5) - torch.pow(batch_y[:,0],5),2))
    b = torch.sum(torch.pow(torch.mean(torch.pow(y[:,0],5))- torch.pow(batch_y[:,0],5),2))
    R20[i,1] = 1 - a/b
    a = torch.pow(torch.sum(torch.pow(y[:,1],5) - torch.pow(batch_y[:,1],5)),2)
    b = torch.sum(torch.pow(torch.mean(torch.pow(y[:,1],5))- torch.pow(batch_y[:,1],5),2))
    R20[i,2] = 1 - a/b
    a = torch.pow(torch.sum(torch.pow(y[:,2],5) - torch.pow(batch_y[:,2],5)),2)
    b = torch.sum(torch.pow(torch.mean(torch.pow(y[:,2],5))- torch.pow(batch_y[:,2],5),2))
    R20[i,0] = 1 - a/b
    
    MSEDATA[i,1] = torch.mean(torch.pow(MaxMin[0]*(torch.pow(y[:,0],5) - torch.pow(batch_y[:,0],5)),2))
    MSEDATA[i,2] = torch.mean(torch.pow(MaxMin[1]*(torch.pow(y[:,1],5) - torch.pow(batch_y[:,1],5)),2))
    MSEDATA[i,0] = torch.mean(torch.pow(MaxMin[2]*(torch.pow(y[:,2],5) - torch.pow(batch_y[:,2],5)),2))
    #(Root Mean Squared Logarithmic Error)
    RMSLE[i,1] =  torch.sqrt(torch.mean(torch.pow( torch.log((MaxMin[0]*torch.pow(y[:,0],5)+1)) - 
                                                  torch.log(MaxMin[0]*torch.pow(batch_y[:,0],5)+1),2)))
    RMSLE[i,2] =  torch.sqrt(torch.mean(torch.pow( torch.log((MaxMin[1]*torch.pow(y[:,1],5)+1)) - 
                                                  torch.log(MaxMin[1]*torch.pow(batch_y[:,1],5)+1),2)))
    RMSLE[i,0] =  torch.sqrt(torch.mean(torch.pow( torch.log((MaxMin[2]*torch.pow(y[:,2],5)+1)) - 
                                                  torch.log(MaxMin[2]*torch.pow(batch_y[:,2],5)+1),2)))
    MAE[i,1] = torch.mean(torch.abs(MaxMin[0]*(torch.pow(y[:,0],5) - torch.pow(batch_y[:,0],5))))
    MAE[i,2] = torch.mean(torch.abs(MaxMin[1]*(torch.pow(y[:,1],5) - torch.pow(batch_y[:,1],5))))
    MAE[i,0] = torch.mean(torch.abs(MaxMin[2]*(torch.pow(y[:,2],5) - torch.pow(batch_y[:,2],5))))    
    a = torch.sum(torch.pow(MaxMin[0]*(torch.pow(y[:,0],5) - torch.pow(batch_y[:,0],5)),2))
    b = torch.sum(torch.pow(MaxMin[0]*(torch.mean(torch.pow(y[:,0],5)) - torch.pow(batch_y[:,0],5)),2))
    R2[i,1] = 1 - a/b
    a = torch.pow(torch.sum(MaxMin[1]*(torch.pow(y[:,1],5) - torch.pow(batch_y[:,1],5))),2)
    b = torch.sum(torch.pow(MaxMin[1]*(torch.mean(torch.pow(y[:,1],5)) - torch.pow(batch_y[:,1],5)),2))
    R2[i,2] = 1 - a/b
    a = torch.pow(torch.sum(MaxMin[2]*(torch.pow(y[:,2],5) - torch.pow(batch_y[:,2],5))),2)
    b = torch.sum(torch.pow(MaxMin[2]*(torch.mean(torch.pow(y[:,2],5)) - torch.pow(batch_y[:,2],5)),2))
    R2[i,0] = 1 - a/b
    
    plt.figure('第二产业经济增长值')
    # plt.subplot(1,3,1)
    plt.title('{}.第二产业经济增长值'.format(Region[i]),FontProperties=font)
    plt.plot(xtime.cpu().numpy(),torch.pow(y[:,0],5).detach().cpu().numpy()*MaxMin[0] + Min[0],'bo-',label='预测值')
    plt.plot(xtime.cpu().numpy(),torch.pow(batch_y[:,0],5).cpu().numpy()*MaxMin[0] + Min[0],'r*-',label='真实值')
    plt.xlabel('年',FontProperties=font)
    plt.ylabel('增长值',FontProperties=font)
    
    plt.legend()
    plt.xticks(xtime,xtime.cpu().numpy().astype(int),rotation = 90)
    # plt.ylim((-0.1,1))
    filename = '.\\figFloder\\{}_第二产业经济增长值.jpg'.format(Region[i])
    plt.savefig(filename)
    plt.close()
   
    # plt.subplot(1,3,2)
    plt.figure('第三产业经济增长值')
    plt.title('{}.第三产业经济增长值'.format(Region[i]),FontProperties=font)
    plt.plot(xtime.cpu().numpy(),torch.pow(y[:,1],5).detach().cpu().numpy()*MaxMin[1] + Min[1],'bo-',label='预测值')
    plt.plot(xtime.cpu().numpy(),torch.pow(batch_y[:,1],5).cpu().numpy()*MaxMin[1] + Min[1],'r*-',label='真实值')
    plt.xlabel('年',FontProperties=font)
    plt.xticks(xtime,xtime.cpu().numpy().astype(int),rotation = 90)
    plt.ylabel('增长值',FontProperties=font)
    plt.legend()
    # plt.ylim((-0.1,1))
    filename = '.\\figFloder\\{}_第三产业经济增长值.jpg'.format(Region[i])
    plt.savefig(filename)
    plt.close()
   
    plt.figure('第一产业经济增长值')
    plt.title('{}.第一产业经济增长值'.format(Region[i]),FontProperties=font)
    plt.plot(xtime.cpu().numpy(),torch.pow(y[:,2],5).detach().cpu().numpy()*MaxMin[2] + Min[2],'bo-',label='预测值')
    plt.plot(xtime.cpu().numpy(),torch.pow(batch_y[:,2],5).cpu().numpy()*MaxMin[2] + Min[2],'r*-',label='真实值')
    plt.xlabel('年',FontProperties=font)
    plt.xticks(xtime,xtime.cpu().numpy().astype(int),rotation = 90)
    plt.ylabel('增长值',FontProperties=font)
    plt.legend()
    filename = '.\\figFloder\\{}_第一产业经济增长值.jpg'.format(Region[i])
    plt.savefig(filename)
    plt.close()
    # plt.ylim((-0.1,1))
    # plt.show()
    
    
    