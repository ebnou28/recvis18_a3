import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
import torchvision.models as models


# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--pretrainedWeights',type=str,default='alexnet-owt-4df8aa71.pth',
                    help='folder where the pretrained Net weights are located.')
parser.add_argument('--network',type=str,default='AlexNet',
                    help="Choose the type of Network [AlexNet or ResNet18 ]")


args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import data_transforms , data_transforms_val 

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms_val),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script

if args.network == 'Net':
    from model import Net
    
    model = Net()
    
elif args.network == 'Net2':
    from model import Net2
    
    model = Net2()
    

elif args.network == "AlexNet":
    from model import AlexNet
    
    model = AlexNet()
    checkpoint = torch.load(args.pretrainedWeights)
    
    model.load_state_dict(checkpoint)
    #reset the last layer 
    fc_numftr = (model.classifier)[-1].in_features
    fc_layer = nn.Linear(fc_numftr, 20)
    (model.classifier)[-1] = fc_layer
    
    
    #Freeze all but last fully connected layer
    i = 0
    for param in (model.parameters()):
        if i < 10 :      #10 layers not to freeze 
            
            param.requires_grad = False
            i+=1
            
elif args.network == "ResNet18":
    from model import ResNet
    from model import BasicBlock


    model = ResNet(BasicBlock, [2, 2, 2, 2])
    
    #fine tune the resnet on our classes
    model.load_state_dict(torch.load(args.pretrainedWeights)) #initialize with the pretrainedweights
    for param in model.parameters():
           param.requires_grad = False

    for param in model.parameters():
           if param.requires_grad == True:
               print("True")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 50)
    #add a final fully connected layer 
    model = nn.Sequential(*[model,nn.ReLU(),nn.Linear(50,20)])
    
    #load the weights of the pretrained layers in their corresponding layers of the model
    #model.load_state_dict(custom_weight_init(model,args.pretrainedWeights))
   
elif args.network == "cResNet18":  
    from model import customResNet
    from model import custom_weight_init
    from model import BasicBlock
    from model import freeze_layers
    
    model = customResNet(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(custom_weight_init(model,args.pretrainedWeights))
    #freeze all but two last 2 layers   
    freeze_layers(model,2)
    
elif args.network == "ResNet152":
    from model import resnet152, BasicBlock, ResNet
    model = resnet152(args.pretrainedWeights,20)
    
    
print(model.modules)
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = args.experiment + '/model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file')




"""
################ two networks training #####################
from model import AlexNet , resnet152, BasicBlock, ResNet 
net = [AlexNet(),resnet152(args.pretrainedWeights,20)]

#loss and optimizer 
parameters = set([])
for net_ in net:
    parameters |= net_.parameters()
    
optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum)    


########## training the networks ##############   
def train(epoch):
    for model in net:
        model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        for model in net:
            out
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
"""