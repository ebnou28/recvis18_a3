import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 20 

import torchvision.models as models
from collections import OrderedDict


class Net(nn.Module):
    

    def __init__(self):
        super(Net, self).__init__()
        
        vgg = models.vgg16_bn(pretrained=True)
        resnet18_ = models.resnet18(pretrained=True)
    
        liste_featresnet = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3'] 
        
        features_resnet = OrderedDict((k , resnet18_._modules[k]) for k in liste_featresnet)
        features_vgg = vgg._modules['features']
            
        self.features_resnet = nn.Sequential(features_resnet)
        self.features_vgg = nn.Sequential(features_vgg)
        
        for par in self.features_vgg.parameters():
            par.requires_grad = False 
            
        for par in self.features_resnet.parameters():
            par.requires_grad = False 
            
        self.resnet_trainable = nn.Sequential(resnet18_._modules['layer4'], resnet18_._modules['avgpool'])
            
        self.fcres = nn.Sequential(nn.Linear(in_features=512, out_features=1024),nn.ReLU(True))
        
        self.fcvgg = nn.Sequential(nn.Linear(25088, 4096),nn.ReLU(True),nn.Dropout(),
                                   nn.Linear(4096, 1024),nn.ReLU(True))

        
        self.classifier = nn.Sequential(nn.Linear(2*1024,512), nn.Linear(512,20))
        
        

    def forward(self, x):

        x1 = self.features_resnet(x)
        x1 = self.resnet_trainable(x1) 
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fcres(x1)
        
        x2 = self.features_vgg(x)
        x2 = x2.view(x2.size(0),-1)
        x2 = self.fcvgg(x2)

        output = torch.cat((x1,x2),-1)
        
        output = self.classifier(output)

        return output
 
class Net2(nn.Module):
    

    def __init__(self):
        super(Net2, self).__init__()
        
        vgg = models.vgg16_bn(pretrained=True)
        resnet152_ = models.resnet152(pretrained=True)
    
        liste_featresnet =['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3'] 
        
        features_resnet = OrderedDict((k , resnet152_._modules[k]) for k in liste_featresnet)
        features_vgg = vgg._modules['features']
            
        self.features_resnet = nn.Sequential(features_resnet)
        self.features_vgg = nn.Sequential(features_vgg)
        
        for par in self.features_vgg.parameters():
            par.requires_grad = False 
            
        for par in self.features_resnet.parameters():
            par.requires_grad = False 
            
        self.resnet_trainable = nn.Sequential(resnet152_._modules['layer4'], resnet152_._modules['avgpool'])
            
        self.fcres = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),nn.ReLU(True))
        
        self.fcvgg = nn.Sequential(nn.Linear(25088, 4096),nn.ReLU(True),nn.Dropout(),
                                   nn.Linear(4096, 1024),nn.ReLU(True))

        
        self.classifier = nn.Sequential(nn.Linear(2*1024,512), nn.Linear(512,20))
        
        

    def forward(self, x):

        x1 = self.features_resnet(x)
        x1 = self.resnet_trainable(x1) 
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fcres(x1)
        
        x2 = self.features_vgg(x)
        x2 = x2.view(x2.size(0),-1)
        x2 = self.fcvgg(x2)

        output = torch.cat((x1,x2),-1)
        
        output = self.classifier(output)

        return output   
    



class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


"""
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
"""

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        
        #### add two more FC-layers  #####
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class customResNet(nn.Module):

    def __init__(self, block, layers, num_classes=20):
        super(customResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        
        #### add two more FC-layers  #####
        
        self.fc1 = nn.Linear(50 * block.expansion, 128)
        self.fc2 = nn.Linear(50, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x



def custom_weight_init(model,pretrWeights,freeze=True):
    """
    Function that takes a network (model) and some weights of some of its layers (that we might
    want to freeze during the training)
    and returns a dict containing the weights for the layers initialized and a random 
    initialization for the other layers
    pretrWeights : path to the weights .pth
    """
    
    #my custom net weight initialization ###
    weight_dict = model.state_dict()
    new_weight_dict = {}  #new dict with the custom weights for freezed and unfreezed layers
    checkpoint = torch.load(pretrWeights) #dictionnary of pretrained weights 
    for param_key in weight_dict.keys():
         # custom initialization in new_weight_dict,
         #initialize the resnet layers that we want to freeze by the pretrained weights
         if param_key in checkpoint and param_key in weight_dict:
             new_weight_dict[param_key] = checkpoint[param_key]  
                 
         #leave the new added layers to train 
         else:
             new_weight_dict[param_key] = weight_dict[param_key]

    
    return weight_dict

def freeze_layers(model,m):
    #m :  m last layers to keep trainable
    
    l = []
    for param in model.parameters():
        if param.requires_grad == True:
            l.append(True)
    n = len(l) 
    
    i = 0
    
    for param in model.parameters():
        if i<len(l)-m:
            param.requires_grad = False
            i+=1
    
    
        

    
#now load the weight custom initialisation to the model 
#model.load_state_dict(custom_weight_init(model,'resnet18-5c106cde.pth'))

pretrWeights = 'resnet152-b121ed2d.pth'

#resnet152
def resnet152(pretrWeights,num_classes, pretrained=True):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(torch.load(pretrWeights))
    num_features = model.fc.in_features    
    model.fc = nn.Linear(num_features,num_classes)
    
    return model


