import numpy as np
from torch.utils import data
import torch
import os
from glob import glob
from utils.utils import read_raw_position
import SimpleITK as sitk
import matplotlib.pyplot as plt

class Dataloader_segm(data.Dataset):
    def __init__(self, split,path_to_data,multiply_files=10,crop_size=128):
        
        self.path_to_data=path_to_data
        self.split=split
        self.crop_size=crop_size
        
        
        names=glob(path_to_data + os.sep + '\**\*_seg.mhd', recursive=True)
        
        
        st0 = np.random.get_state()
        np.random.seed(seed=42)
        names=[names[i] for i in np.random.permutation(len(names))]
        np.random.set_state(st0)
        
        
        split_fraction=0.8
        position=int(split_fraction*len(names))
        
        if self.split=='train':   
            names=names[:position]
        elif self.split=='test':
            names=names[position:]
        else:
            raise Exception('train or test')
                
        
        
        
        self.file_names=names
        
        self.sizes=[]
        for name in self.file_names:
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(name)
            file_reader.ReadImageInformation()
            self.sizes.append(file_reader.GetSize())
        
        
        self.file_names=self.file_names*multiply_files
        self.sizes=self.sizes*multiply_files
        
        
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        
        file_name_seg=self.file_names[index]
        file_name=file_name_seg.replace('_seg','')
        
        file_name_gt=file_name_seg
        # file_name_gt=file_name_seg.replace('_seg','_dt')
        
        size=self.sizes[index]
        
        crop_size=int(self.crop_size)
        
        p=-1*np.ones(3)
        if (size[0]-crop_size)>0:
            p[0]=torch.randint(size[0]-crop_size,(1,1)).view(-1).numpy()[0]
        if (size[1]-crop_size)>0:
            p[1]=torch.randint(size[1]-crop_size,(1,1)).view(-1).numpy()[0]
        if (size[2]-crop_size)>0:
            p[2]=torch.randint(size[2]-crop_size,(1,1)).view(-1).numpy()[0]
 
        
        img = read_raw_position(file_name,[crop_size,crop_size,crop_size],[int(p[0]),int(p[1]),int(p[2])])
        
        gt = read_raw_position(file_name_gt,[crop_size,crop_size,crop_size],[int(p[0]),int(p[1]),int(p[2])])
    
        
    
        gt=gt>0
        
        
        img=img.astype(np.float32)
        img=np.expand_dims(img, axis=0).copy()
        img=torch.from_numpy(img)
        
        gt=gt.astype(np.float32)
        gt=np.expand_dims(gt, axis=0).copy()
        gt=torch.from_numpy(gt)
        
       
        return img,gt
        
        
        
        
    
    
        
if __name__ == '__main__':
    
    
    loader = Dataloader_segm(split='train',path_to_data=r'Z:\CELL_MUNI\verse2020\training_data_resaved')
    trainloader= data.DataLoader(loader, batch_size=2, num_workers=0, shuffle=True,drop_last=True)


    for i,(img,mask) in enumerate(trainloader):
        
        img_np=img.detach().cpu().numpy()
        mask_np=mask.detach().cpu().numpy()
        dt_np=dt.detach().cpu().numpy()
        
        plt.imshow(np.max(img_np[0,0,:,:,:],axis=0))
        plt.show()
        plt.imshow(np.max(mask_np[0,0,:,:,:],axis=0))
        plt.show()
        plt.imshow(np.max(mask_np[0,1,:,:,:],axis=0))
        plt.show()
        
        plt.imshow(np.max(img_np[0,0,:,:,:],axis=1))
        plt.show()
        plt.imshow(np.max(mask_np[0,0,:,:,:],axis=1))
        plt.show()
        plt.imshow(np.max(mask_np[0,1,:,:,:],axis=1))
        plt.show()
        
        
        plt.imshow(np.max(img_np[0,0,:,:,:],axis=2))
        plt.show()
        plt.imshow(np.max(mask_np[0,0,:,:,:],axis=2))
        plt.show()
        plt.imshow(np.max(mask_np[0,1,:,:,:],axis=2))
        plt.show()
        
        
        break
        
        
        
        
        
        