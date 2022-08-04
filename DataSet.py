from math import isnan
import os
import cv2
import numpy as np
import json
import time

import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
from torch.utils import data
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from torch.autograd import Variable



def tonemapper(exr,mode=0):
    if mode ==0:
        return torch.pow(torch.clamp(exr,0.0,1.0), (1/2.2))
    elif mode == 1:
        A = 2.51
        B = 0.03
        C = 2.43
        D = 0.59
        E = 0.14
        return torch.pow(torch.clamp(( (exr * (A * exr + B)) / (exr * (C * exr + D) + E) ),0.0,1.0), (1/2.2))

def probe_img2channel(probe_hdr, env_h, pano_h, pano_w):
    """translate probe tensor 2 channel tensor

    Args:
        probe_hdr ([type]): [shape: (3,env_h*pano_h, env_w*pano_w)]
        env_h ([type]): [height of every probe]
        pano_h ([type]): [height of rendered panorama]

    Returns:
        [type]: [shape: (3*env_h*env_w, pano_h, pano_w)]
    """
    c,h,w = probe_hdr.shape
    probe_hdr = probe_hdr.reshape(3, pano_h, env_h, pano_w, env_h*2)
    probe_hdr = probe_hdr.permute(0,2,4,1,3).reshape(-1, pano_h, pano_w)

    return probe_hdr

def probe_channel2img(probe_channels, env_h, pano_h, pano_w):
    """translate channel tensor 2 probe tensor

    Args:
        probe_channels ([type]): [shape: (3*env_h*env_w, pano_h, pano_w)]
        env_h ([type]): [height of every probe]
        pano_h ([type]): [height of rendered panorama]

    Returns:
        [type]: [shape: (3,env_h*pano_h, env_w*pano_w)]
    """
    c,h,w = probe_channels.shape
    probe_channels = probe_channels.reshape(3,env_h, int(env_h*2), pano_h, pano_w)
    probe_channels = probe_channels.permute(0,3,1,4,2).reshape(3, env_h*pano_h, int(env_h*2)*pano_w)

    return probe_channels


class KePanoMaterial(Dataset):
    r"""read pano data,
    it's format is .hdr.
    example['image'],
    example['albedo'],
    example['normal'],
    example['roughness'],
    example['metallic'],
    example['depth'],
    example['mask']
    """
    def __init__(self,root,cube_lenth=128,pano_height=256,pano_width=512,split_json_path="",mode="train",is_random_exposure=True) -> None:
        super().__init__()
        
        self.root = root
        self.cube_lenth = cube_lenth
        self.pano_height = pano_height
        self.pano_width = pano_width

        self.max_depth = 10   # 10 m, norm [0,1]


        self.split_json_path = split_json_path
        self.mode = mode

        self.all_item = self.read_all_item(root)
        self.is_random_exposure = is_random_exposure

    def __getitem__(self, index):
        one_item = self.all_item[index]
        one_path = one_item[0]
        iindex = int(one_item[1])
        #print("{} + index {}".format(one_path,iindex))

        
        # range: [-2,-0.5)
        if self.is_random_exposure:
            random_exposure = torch.rand(1)*1.5 - 2.0
        else:
            random_exposure = -1.0

        image = cv2.imread(os.path.join(one_path,str(iindex)+'_image.hdr'),-1)[:,:,0:3]
        image = cv2.resize(image,(self.pano_width,self.pano_height))
        image = np.asarray(image,dtype=np.float32)
        image = image[...,::-1].copy()
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        image = image * torch.pow(torch.tensor(2.0),random_exposure)
        image = tonemapper(image,mode=1)    # ACES tonemapping

        albedo = cv2.imread(os.path.join(one_path,str(iindex)+'_albedo.hdr'),-1)[:,:,0:3]
        albedo = cv2.resize(albedo,(self.pano_width,self.pano_height))
        albedo = np.asarray(albedo,dtype=np.float32)
        albedo = albedo[...,::-1].copy()
        albedo = torch.from_numpy(albedo)
        albedo = albedo.permute(2,0,1)

        roughness = cv2.imread(os.path.join(one_path,str(iindex)+'_roughness.hdr'),-1)[:,:,0:1]
        roughness = cv2.resize(roughness,(self.pano_width,self.pano_height))
        roughness = np.asarray(roughness,dtype=np.float32)
        roughness = torch.from_numpy(roughness)
        roughness = roughness.unsqueeze(0)

        metallic = cv2.imread(os.path.join(one_path,str(iindex)+'_metallic.hdr'),-1)[:,:,0:1]
        metallic = cv2.resize(metallic,(self.pano_width,self.pano_height))
        metallic = np.asarray(metallic,dtype=np.float32)
        metallic = torch.from_numpy(metallic)
        metallic = metallic.unsqueeze(0)

        mask = cv2.imread(os.path.join(one_path,str(iindex)+'_mask.hdr'),-1)[:,:,0:1]
        mask = cv2.resize(mask,(self.pano_width,self.pano_height))
        mask = np.asarray(mask,dtype=np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)

        normal = cv2.imread(os.path.join(one_path,str(iindex)+'_normal.hdr'),-1)[:,:,0:3]
        normal = cv2.resize(normal,(self.pano_width,self.pano_height))
        normal = np.asarray(normal,dtype=np.float32)
        normal = normal[...,::-1].copy()
        normal = (normal*2.0)-1.0
        normal = torch.from_numpy(normal)
        normal = normal.permute(2,0,1)

        depth = cv2.imread(os.path.join(one_path,str(iindex)+'_depth.hdr'),-1)[:,:,0:1]
        depth = cv2.resize(depth,(self.pano_width,self.pano_height))
        depth = np.asarray(depth,dtype=np.float32)
        depth = torch.from_numpy(depth)
        depth = depth.unsqueeze(0)
        depth = depth * torch.tensor(self.max_depth)/2

        depth_mask = (depth>0) & (depth<=self.max_depth) & (~torch.isnan(depth))

        
        

        #print("total time cost:{}".format(time.time()-start_time))
        name = one_path.split('/')[-3]+"_"+one_item[1]
        
        batchDict = {'image':image,
            'albedo':albedo,
            'normal':normal,
            'roughness':roughness,
            'metallic':metallic,
            'depth':depth,
            'depth_mask':depth_mask,
            'mask':mask,
            'name':name}
        
        return batchDict

    def __len__(self):
        return len(self.all_item)

    def read_all_item(self,root):
        all_item = []
        with open(self.split_json_path,'r') as f:
            strs = f.read()
        split_json = json.loads(strs)
        planids = split_json[self.mode]
        for id in planids:
            if not os.path.exists(os.path.join(root,id)):
                continue
            whole_path = os.path.join(root,id,'ue4_result','CubemapData')
            items = os.listdir(whole_path)
            #print(id)
            # 5--without normal and depth, 6--with normal, 7 --with depth
            #assert (len(items)%(6))==0
            if (len(items)%(7))!=0:
                print(id)
                continue
            num = round(len(items)/7)
            for i in range(0,num):
                one_item = []
                one_item.append(whole_path)
                one_item.append(str(i))
                all_item.append(one_item)
                one_item = []
        print(len(all_item))
        return all_item


class KePanoLighting(Dataset):
    r"""read channel cube data,
    it's format is .npz.
    example['image'],
    example['albedo'],
    example['normal'],
    example['roughness'],
    example['metallic'],
    example['depth'],
    example['mask']
    """
    def __init__(self,root,cube_lenth=128,pano_height=256,pano_width=512,env_h=16,probes_h=128,split_json_path="",mode="train",is_random_exposure=True) -> None:
        super().__init__()
        
        self.root = root
        self.cube_lenth = cube_lenth
        self.pano_height = pano_height
        self.pano_width = pano_width
        self.probes_h = probes_h
        self.probes_w = int(probes_h*2)
        self.env_h = env_h
        self.env_w = int(env_h*2)

        self.max_depth = 10   # 10 m, norm [0,1]


        self.split_json_path = split_json_path
        self.mode = mode

        self.all_item = self.read_all_item(root)
        self.is_random_exposure = is_random_exposure

    def __getitem__(self, index):
        one_item = self.all_item[index]
        one_path = one_item[0]
        iindex = int(one_item[1])
        #print("{} + index {}".format(one_path,iindex))

        
        # range: [-2,-0.5)
        if self.is_random_exposure:
            random_exposure = torch.rand(1)*1.5 - 2.0
        else:
            random_exposure = -1.0
        
    
        try:
            light = cv2.imread(os.path.join(one_path,str(iindex)+'_light.exr'),-1)[:,:,0:3]
        except:
            print(os.path.join(one_path,str(iindex)+'_light.exr'))
        
        light = np.asarray(light,dtype=np.float32)
        light = light[...,::-1].copy()
        light = torch.from_numpy(light)
        light = light.permute(2,0,1)
        light = probe_img2channel(light,self.env_h,self.probes_h,self.probes_w)
        light = light * torch.pow(torch.tensor(2.0),random_exposure)
        
        one_path = one_path.replace('LightProbeData','CubemapData').replace('KePanoLight','KePanoData')

        image = cv2.imread(os.path.join(one_path,str(iindex)+'_image.hdr'),-1)[:,:,0:3]
        image = cv2.resize(image,(self.pano_width,self.pano_height))
        image = np.asarray(image,dtype=np.float32)
        image = image[...,::-1].copy()
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        image = image * torch.pow(torch.tensor(2.0),random_exposure)
        image = tonemapper(image,mode=1)    # ACES tonemapping

        albedo = cv2.imread(os.path.join(one_path,str(iindex)+'_albedo.hdr'),-1)[:,:,0:3]
        albedo = cv2.resize(albedo,(self.pano_width,self.pano_height))
        albedo = np.asarray(albedo,dtype=np.float32)
        albedo = albedo[...,::-1].copy()
        albedo = torch.from_numpy(albedo)
        albedo = albedo.permute(2,0,1)

        roughness = cv2.imread(os.path.join(one_path,str(iindex)+'_roughness.hdr'),-1)[:,:,0:1]
        roughness = cv2.resize(roughness,(self.pano_width,self.pano_height))
        roughness = np.asarray(roughness,dtype=np.float32)
        roughness = torch.from_numpy(roughness)
        roughness = roughness.unsqueeze(0)

        metallic = cv2.imread(os.path.join(one_path,str(iindex)+'_metallic.hdr'),-1)[:,:,0:1]
        metallic = cv2.resize(metallic,(self.pano_width,self.pano_height))
        metallic = np.asarray(metallic,dtype=np.float32)
        metallic = torch.from_numpy(metallic)
        metallic = metallic.unsqueeze(0)

        mask = cv2.imread(os.path.join(one_path,str(iindex)+'_mask.hdr'),-1)[:,:,0:1]
        mask = cv2.resize(mask,(self.pano_width,self.pano_height))
        mask = np.asarray(mask,dtype=np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)

        normal = cv2.imread(os.path.join(one_path,str(iindex)+'_normal.hdr'),-1)[:,:,0:3]
        normal = cv2.resize(normal,(self.pano_width,self.pano_height))
        normal = np.asarray(normal,dtype=np.float32)
        normal = normal[...,::-1].copy()
        normal = (normal*2.0)-1.0
        normal = torch.from_numpy(normal)
        normal = normal.permute(2,0,1)

        depth = cv2.imread(os.path.join(one_path,str(iindex)+'_depth.hdr'),-1)[:,:,0:1]
        depth = cv2.resize(depth,(self.pano_width,self.pano_height))
        depth = np.asarray(depth,dtype=np.float32)
        depth = torch.from_numpy(depth)
        depth = depth.unsqueeze(0)
        depth = depth * torch.tensor(self.max_depth)/2

        depth_mask = (depth>0) & (depth<=self.max_depth) & (~torch.isnan(depth))

        
        

        #print("total time cost:{}".format(time.time()-start_time))
        name = one_path.split('/')[-3]+"_"+one_item[1]
        
        batchDict = {'image':image,
            'albedo':albedo,
            'normal':normal,
            'roughness':roughness,
            'metallic':metallic,
            'depth':depth,
            'depth_mask':depth_mask,
            'mask':mask,
            'light':light,
            'name':name}
        
        return batchDict

    def __len__(self):
        return len(self.all_item)

    def read_all_item(self,root):
        all_item = []
        with open(self.split_json_path,'r') as f:
            strs = f.read()
        split_json = json.loads(strs)
        planids = split_json[self.mode]
        for id in planids:
            if not os.path.exists(os.path.join(root,id)):
                continue
            if not os.path.exists(os.path.join(root.replace('KePanoLight','KePanoData'),id)):
                continue
            whole_path = os.path.join(root,id,'ue4_result','LightProbeData')
            items = os.listdir(whole_path)
            #print(id)
            # 5--without normal and depth, 6--with normal, 7 --with depth
            #assert (len(items)%(6))==0
            if (len(items))!=1:
                print(id)
                continue
            num = len(items)
            for i in range(0,num):
                one_item = []
                one_item.append(whole_path)
                one_item.append(str(i))
                all_item.append(one_item)
                one_item = []
        print(len(all_item))
        return all_item

