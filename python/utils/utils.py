
import numpy as np
import SimpleITK as sitk

def rotate_3d(img,rot_axes):
    
    a,b,c = rot_axes
    
    img=np.rot90(img,int(a),axes=(1,2))
    img=np.rot90(img,int(b),axes=(0,2))
    img=np.rot90(img,int(c),axes=(0,1))
    
    return img

def rotate_3d_inverse(img,rot_axes):
    
    a,b,c = rot_axes
    a,b,c = -int(a),-int(b),-int(c)
    a,b,c=a%4,b%4,c%4
    
    
    img=np.rot90(img,c,axes=(0,1))
    img=np.rot90(img,b,axes=(0,2))
    img=np.rot90(img,a,axes=(1,2))
    
    return img

def read_raw_position(file_name,extract_size,current_index):
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)
    
    file_reader.ReadImageInformation()
    size=file_reader.GetSize()    
    
    img=np.zeros(extract_size,dtype=np.float32)
    
    for k in range(3):
        if current_index[k]==-1:
            current_index[k]=0
            extract_size[k]=size[k]
    
    file_reader.SetExtractIndex(current_index)
    file_reader.SetExtractSize(extract_size)
    
    
    
    tmp=sitk.GetArrayFromImage(file_reader.Execute())
    
    img[:tmp.shape[0],:tmp.shape[1],:tmp.shape[2]]=tmp
    
    return img




import torch

def dice_loss(pred, target):
  
    smooth = 1.
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )




def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']