import sys
import os
import numpy as np
import random
import time
import math
import csv
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import argmax
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms.functional as Ft
from torch.autograd.variable import Variable

from scipy.ndimage.morphology import distance_transform_edt as dist_trans

# MISC FUNCTIONS #

""" 
    Export data to csv format. Creates new file if doesn't exist,
    otherwise update it.
    Args:
        header (list): headers of the column
        value (list): values of correspoding column
        folder: folder path
        file_name: file name with path
"""
def export_history(header, value, folder, file_name):
    # If folder does not exists make folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = folder + file_name
    file_existence = os.path.isfile(file_path)

    # If there is no file make file
    if file_existence == False:
        file = open(file_path, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(value)
    # If there is file overwrite
    else:
        file = open(file_path, 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(value)
    # Close file when it is done with writing
    file.close()


""" 
    Save the state of a net.
"""
def save_checkpoint(state, path='checkpoint/', filename='weights.pth'):
    # If folder does not exists make folder
    if not os.path.exists(path):
        os.makedirs(path)

    filepath = os.path.join(path, filename)
    torch.save(state, filepath)


""" 
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
"""  
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# DEFINITION OF THE ARCHITECTURE #

""" 
    This file defines every layer (or group of layers) that are inside UNet.
    At the final the architecture UNet is defined as a conjuntion of the elements created.
"""
class double_conv(nn.Module):
    ''' Applies (conv => BN => ReLU) two times. '''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            # inplace is for aply ReLU to the original place, saving memory
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            # inplace is for aply ReLU to the original place, saving memory
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    ''' First Section of U-Net. '''

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    ''' Applies a MaxPool with a Kernel of 2x2,
        then applies a double convolution pack. '''

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    ''' Applies a Deconvolution and then applies applies a double convolution pack. '''

    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        
        # Bilinear is used to save computational cost
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_ch//2, in_ch//2, kernel_size=2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(input=x2, pad=(diffX // 2, diffX // 2,
                                  diffY // 2, diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    ''' Applies the last Convolution to give an answer. '''

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    ''' This Object defines the architecture of U-Net. '''

    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        #x = F.softmax(x) # New softmax layer
        return x

# FUNCTIONS FOR LOADING THE DATA #

'''
    Class that defines the reading and processing of the images.
    Specialized on BBBC006 dataset.
'''
class BBBCDataset(Dataset):
    def __init__(self, ids, dir_data, dir_gt, extension='.png', isWCE=False, dir_weights = ''):

        self.dir_data = dir_data
        self.dir_gt = dir_gt
        self.extension = extension
        self.isWCE = isWCE
        self.dir_weights = dir_weights

        # Transforms
        self.data_transforms = {
            'imgs': transforms.Compose([
#                 transforms.RandomResizedCrop(256),
#                 transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
#                transforms.Normalize([0.0054],[0.0037])
            ]),
            'masks': transforms.Compose([
                transforms.ToTensor()
            ]),
        }

        # Images IDS
        self.ids = ids

        # Calculate len of data
        self.data_len = len(self.ids)

    '''
        Ask for an image.
    '''
    def __getitem__(self, index):
        # Get an ID of a specific image
        id_img = self.dir_data + self.ids[index] + self.extension
        id_gt = self.dir_gt + self.ids[index] + self.extension
        # Open Image and GroundTruth
        img = Image.open(id_img).convert('L')
        gt = Image.open(id_gt)
        # Applies transformations
        img = self.data_transforms['imgs'](img)
        gt = self.data_transforms['masks'](gt)
        if self.isWCE:
            id_weight = self.dir_weights + self.ids[index] + self.extension
            weight = Image.open(id_weight).convert('L')
            weight = self.data_transforms['masks'](weight)
            return (img, gt, weight)

        return (img, gt, gt)

    '''
        Length of the dataset.
        It's needed for the upper class.
    '''
    def __len__(self):
        return self.data_len

##import IPython
##js_code = '''
##document.querySelector("#output-area").appendChild(document.createTextNode("hello world!"));
##'''
##display(IPython.display.Javascript(js_code))

'''
    Class that defines the reading and processing of the images.
    Specialized on BBBC006 dataset.
'''
class BBBCDataset_Transform(Dataset):
    def __init__(self, ids, dir_data, extension='.png'):

        self.dir_data = dir_data
        self.extension = extension

        # Transforms
        self.data_transforms = {
            'imgs': transforms.Compose([
#                 transforms.RandomResizedCrop(256),
#                 transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize([0.0054],[0.0037])
            ]),
            'masks': transforms.Compose([
                transforms.ToTensor()
            ]),
        }

        # Images IDS
        self.ids = ids

        # Calculate len of data
        self.data_len = len(self.ids)

    '''
        Ask for an image.
    '''
    def __getitem__(self, index):
        # Get an ID of a specific image
        id_img = self.dir_data + self.ids[index] + self.extension
        # Open Image and GroundTruth
        img = Image.open(id_img).convert('L')
        # Applies transformations
        img = self.data_transforms['imgs'](img)
        return (img, self.ids[index]+self.extension)

    '''
        Length of the dataset.
        It's needed for the upper class.
    '''
    def __len__(self):
        return self.data_len

'''
    Returns the dataset separated in batches.
    Used inside every epoch for retrieving the images.
'''
def get_dataloaders(dir_img, dir_gt, test_percent=0.2, batch_size=10, isWCE = False, dir_weights=''):
    # Validate a correct percentage
    test_percent = test_percent/100 if test_percent > 1 else test_percent
    # Read the names of the images
    ids = [f[:-4] for f in os.listdir(dir_img)]
    # Creates the dataset
    if not isWCE:
        dset = BBBCDataset(ids, dir_img, dir_gt)
    else:
        dset = BBBCDataset(ids, dir_img, dir_gt, isWCE = isWCE, dir_weights = dir_weights)
    
    # Calculate separation index for training and validation
    num_train = len(dset)
    indices = list(range(num_train))
    split = int(np.floor(test_percent * num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    # Create the dataloaders
    dataloaders = {}
    dataloaders['train'] = DataLoader(dset, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_idx))
    dataloaders['val'] = DataLoader(dset, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_idx))
   
    return dataloaders['train'], dataloaders['val']

'''
    Returns few images for showing the results.
'''
def get_dataloader_show(dir_img, dir_gt):
    # Read the names of the images
    ids = [f[:-4] for f in os.listdir(dir_img)]
    # Creates the dataset
    dset = BBBCDataset(ids, dir_img, dir_gt)

    # Create the dataloader
    dataloader = DataLoader(dset, batch_size=len(ids))
   
    return dataloader

'''
    Returns whole dataset to transform.
'''
def get_dataloader_transform(dir_img, batch_size = 1):
    # Read the names of the images
    ids = [f[:-4] for f in os.listdir(dir_img)]
    # Creates the dataset
    dset = BBBCDataset_Transform(ids, dir_img)

    # Create the dataloader
    dataloader = DataLoader(dset, batch_size=batch_size)

    return dataloader

# IMAGE FROM TENSOR #
def to_image(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    return im

# SEEING THE RESULTS #

def predict_imgs(net, device, loader, criterion, show=False, post=None, dest=""):
    
    #net.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    i = 1
    
    with torch.no_grad():
        for batch_idx, (data, gt, id) in enumerate(loader):
   
            # Use GPU or not
            data, gt = data.to(device, dtype=torch.float), gt.to(device, dtype=torch.float)
            
            fig=plt.figure(figsize=(20, 20))
            
            '''if show:
                # Shows original image
                data_img = transforms.ToPILImage()(data.squeeze(0).cpu()).convert('RGB')
                fig=plt.figure(figsize=(20, 20))
                fig.add_subplot(1, 4, 1)
                plt.imshow(data_img)
            '''

            # Forward
            predictions = net(data)
            
            if post == "Softmax":
              predictions = F.softmax(predictions)
            
            save_image(predictions, dest+"/res_" + str(i) + '.png')
            #save_image(gt, 'results/gt_' + str(i) + '.png')
            
            # Loss Calculation
            loss = criterion(predictions, gt)

            # Updates the record
            val_loss.update(loss.item(), predictions.size(0))
            val_acc.update(-loss.item(), predictions.size(0))
            
            predictions = predictions.squeeze(0).cpu()
            
            '''
            # Shows prediction
            if show:
                # Shows original image
                ori = to_image(data)
                fig.add_subplot(1,4,1)
                plt.imshow(ori)
                # Shows prediction probability
                pred_p = to_image(predictions).convert('RGB')
                fig.add_subplot(1, 4, 2)
                #fig.add_subplot(1,4,1)
                plt.imshow(pred_p)
                # Shows gt
                gt_img = transforms.ToPILImage()(gt.squeeze(0).cpu()).convert('RGB')
                fig.add_subplot(1, 4, 3)
                #fig.add_subplot(1, 4, 2)
                plt.imshow(gt_img)
                plt.show()
            '''
                
            # Prints the loss value of every image proccessed    
            print('[{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(loader),
                100. * batch_idx / len(loader), loss.item()))
                
            i += 1
            
    print('\nValidation set: Average loss: '+ str(val_loss.avg))
    print('\nAverage Validation Accuracy: ' + str(val_acc.avg))

def get_predloader(dir_img, dir_img_2, dir_gt, batch_size=1):
    # Read the names of the images
    ids = [f[:-4] for f in os.listdir(dir_img)]
    # Rearrange the images
    random.shuffle(ids)
    # Calculate index of partition
    ids_pred = ids[:10]

    # Create the datasets
    if dir_img_2 != None:
      pred_dataset = BBBCDataset_2_inputs(ids=ids_pred, dir_data_1=dir_img, dir_data_2=dir_img_2, dir_gt=dir_gt)
    else:
      pred_dataset = BBBCDataset(ids=ids_pred, dir_data=dir_img, dir_gt=dir_gt)

    # Create the loaders
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=True)

    return pred_loader

def predict(load="checkpoints/Border/weights1.pth", n_channels=1, n_classes=1, dir_pred="data/Original/Test/", dir_pred_2="predictions/DT/MAE/Test/", dir_gt="data/Distance_Transform/Test/", evaluation="MSE", out=None, dest="C:/a/"):

    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    print("HAY CUDA:", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device? : ", device)
    # Create the model - 256 classes are used so the resulting image can better reflect the grayscale groundtruth
    #net = UNet(n_channels=1, n_classes=1).to(device)
    # Create the model
    net = UNet(n_channels=n_channels, n_classes = n_classes).to(device)
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).to(device)

    #net = Unet(n_classes = 1)
    #print("Load: ", load)
    # Load old weights
    if load:
        net.load_state_dict(torch.load(load, map_location="cpu")["state_dict"])
        print('Model loaded from {}'.format(load))
    

    # Definition of the evaluation function
    if evaluation == "Dice":
        criterion_val = DiceLoss()
    elif evaluation == "RMSE":
        criterion_val = RMSELoss()
    elif evaluation == "MSE":
        criterion_val = nn.MSELoss()
    elif evaluation == "MAE":
        criterion_val = nn.L1Loss()
    
    pred_loader = get_predloader(dir_pred, dir_pred_2, dir_gt)

    # Run the prediction
    predict_imgs(net=net,
                device=device,
                loader=pred_loader,
                criterion = criterion_val,
                show=True,
                post=out,
                dest=dest)
                
def predict_images2(net, device, loader, criterion, show=False, post=None, dest=""):
    
    i = 1
    
    with torch.no_grad():
        for batch_idx, (data, gt, id) in enumerate(loader):
   
            # Use GPU or not
            data, gt = data.to(device, dtype=torch.float), gt.to(device, dtype=torch.float)
            
            fig=plt.figure(figsize=(20, 20))
            
            '''if show:
                # Shows original image
                data_img = transforms.ToPILImage()(data.squeeze(0).cpu()).convert('RGB')
                fig=plt.figure(figsize=(20, 20))
                fig.add_subplot(1, 4, 1)
                plt.imshow(data_img)
            '''

            # Forward
            predictions = net(data)
            
            if post == "Softmax":
              predictions = F.softmax(predictions)
            
            save_image(predictions, dest+"/res_" + str(i) + '.png')
            #save_image(gt, 'results/gt_' + str(i) + '.png')
            
            # Loss Calculation
            #loss = criterion(predictions, gt)

            # Updates the record
            #val_loss.update(loss.item(), predictions.size(0))
            #val_acc.update(-loss.item(), predictions.size(0))
            
            #predictions = predictions.squeeze(0).cpu()
            
            '''
            # Shows prediction
            if show:
                # Shows original image
                ori = to_image(data)
                fig.add_subplot(1,4,1)
                plt.imshow(ori)
                # Shows prediction probability
                pred_p = to_image(predictions).convert('RGB')
                fig.add_subplot(1, 4, 2)
                #fig.add_subplot(1,4,1)
                plt.imshow(pred_p)
                # Shows gt
                gt_img = transforms.ToPILImage()(gt.squeeze(0).cpu()).convert('RGB')
                fig.add_subplot(1, 4, 3)
                #fig.add_subplot(1, 4, 2)
                plt.imshow(gt_img)
                plt.show()
            '''
                
            i += 1
           
def predict2(load="checkpoints/Border/weights1.pth", n_channels=1, n_classes=1, dir_pred="data/Original/Test/", dir_pred_2="predictions/DT/MAE/Test/", dir_gt="data/Distance_Transform/Test/", evaluation="MSE", out=None, dest="C:/a/"):

    # Use GPU or not
    use_cuda = torch.cuda.is_available()
    print("HAY CUDA:", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device? : ", device)
    # Create the model - 256 classes are used so the resulting image can better reflect the grayscale groundtruth
    #net = UNet(n_channels=1, n_classes=1).to(device)
    # Create the model
    net = UNet(n_channels=n_channels, n_classes = n_classes).to(device)
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).to(device)

    #net = Unet(n_classes = 1)
    #print("Load: ", load)
    # Load old weights
    if load:
        net.load_state_dict(torch.load(load, map_location="cpu")["state_dict"])
        print('Model loaded from {}'.format(load))
    

    # Definition of the evaluation function
    #if evaluation == "Dice":
    criterion_val = nn.MSELoss()

    
    pred_loader = get_predloader(dir_pred, dir_pred_2, dir_gt)

    # Run the prediction
    predict_images2(net=net,
                device=device,
                loader=pred_loader,
                criterion = criterion_val,
                show=True,
                post=out,
                dest=dest)

# TEST PREDICTIONS FOR BACK MODEL #

'''predict(load='C:\\Users\\pc\\Desktop\\weights_CET.pth',
        n_channels = 1,
        n_classes = 3,
        dir_pred = 'C:\\Users\\pc\\Desktop\\Imagenes\\',
        dir_pred_2 = None,
        dir_gt = 'C:\\Users\\pc\\Desktop\\Imagenes\\',
        evaluation="MSE", 
        out=None)

# Test Predictions for top model #
predict(load='C:\\Users\\pc\\Desktop\\weights_CET.pth',
        n_channels = 1,
        n_classes = 3,
        dir_pred = 'C:\\Users\\pc\\Desktop\\Imagenes\\',
        dir_pred_2 = None,
        dir_gt = 'C:\\Users\\pc\\Desktop\\Imagenes\\',
        evaluation="MSE", 
        out=None)
'''

predict2(load="../weights/weights0.pth",
        n_channels = 1,
        n_classes = 3,
        dir_pred = "../pesos/DataSet_39/",
        dir_pred_2 = None,
        dir_gt = "../pesos/DataSet_39/",
        evaluation="MSE", 
        out=None,
dest="../pesos/DataSet_39_resultado2/")
