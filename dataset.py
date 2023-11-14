from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img
import h5py
import io
import sparse
import imageio.v2 as imageio
import torchvision.transforms.functional as TF
import os
import numpy as np

# class DatasetFromFolder(data.Dataset):
#     def __init__(self, image_dir, direction):
#         super(DatasetFromFolder, self).__init__()
#         self.direction = direction
#         self.a_path = join(image_dir, "a")
#         self.b_path = join(image_dir, "b")
#         self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

#         transform_list = [transforms.ToTensor(),
#                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

#         self.transform = transforms.Compose(transform_list)

#     def __getitem__(self, index):
#         a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
#         b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
#         a = a.resize((286, 286), Image.BICUBIC)
#         b = b.resize((286, 286), Image.BICUBIC)
#         a = transforms.ToTensor()(a)
#         b = transforms.ToTensor()(b)
#         w_offset = random.randint(0, max(0, 286 - 256 - 1))
#         h_offset = random.randint(0, max(0, 286 - 256 - 1))
    
#         a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
#         b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
#         a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
#         b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

#         if random.random() < 0.5:
#             idx = [i for i in range(a.size(2) - 1, -1, -1)]
#             idx = torch.LongTensor(idx)
#             a = a.index_select(2, idx)
#             b = b.index_select(2, idx)

#         if self.direction == "a2b":
#             return a, b
#         else:
#             return b, a

#     def __len__(self):
#         return len(self.image_filenames)


# class ImageDataset(Dataset):
#     def __init__(self, root, transforms_=None, mode="train"):
#         self.transform = transforms.Compose(transforms_)

#         self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
#         if mode == "train":
#             self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

#     def __getitem__(self, index):

#         img = Image.open(self.files[index % len(self.files)])
#         w, h = img.size
#         img_A = img.crop((0, 0, w / 2, h))
#         img_B = img.crop((w / 2, 0, w, h))

#         if np.random.random() < 0.5:
#             img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
#             img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

#         img_A = self.transform(img_A)
#         img_B = self.transform(img_B)

#         return {"A": img_A, "B": img_B}

#     def __len__(self):
#         return len(self.files)

def imread(path):
  image = imageio.imread(path)
  return image
def preprocess(image):

  image = image / 255.0
  #image = np.where(image >= 0, image, 0)
  return image




# class brooklynqueensdataset():

#     def __init__(self,mode):
#     #   BaseDataset.__init__(self, opt)
#       self.dirA = os.path.join('/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens','brooklyn/overhead/images/19') # phase 另算
#       self.dirB = os.path.join('/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens','brooklyn/streetview/images')
      
#       #self.dir_seg = os.path.join()
#       name = 'brooklyn-fc8_landuse'
#       neighbors=20
#       #import pdb; pdb.set_trace()
#       if any(x in name for x in ['brooklyn', 'queens']):
      
#         name, label = name.split('_')    # brooklyn-fc8  landcover
#         local_name = name.split('-')[0]
#         #import pdb; pdb.set_trace()
#         data_dir = '/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens/' #"/u/eag-d1/data/near-remote/"
#         self.aerial_dir = "{}{}/overhead/".format(data_dir, local_name)  #"{}{}/aerial/".format(data_dir, local_name)
#         self.label_dir = "{}{}/labels/{}/".format(data_dir, local_name, label)
#         self.streetview_dir = "{}{}/streetview/".format(data_dir, local_name)
#         # self.streetview_seg_dir = "{}{}/streetview/seg_fast/".format(data_dir, local_name)
#       else:
#         raise ValueError('Unknown dataset.')
      
#       self.name = name
#       self.label = label
#       self.base_dir = "/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens/"  #"/u/eag-d1/scratch/scott/learn/near-remote/data/"
#       self.config = self.setup(name, label, neighbors)
      
#       #import pdb; pdb.set_trace()
#       self.h5_name = "{}_train.h5".format(name) if mode in [
#           "train", "val"
#       ] else "{}_test.h5".format(name)

#       tmp_h5 = h5py.File("{}{}/{}".format(self.base_dir, self.name, self.h5_name),
#                         'r')
#       self.dataset_len = len(tmp_h5['fname'])
#       self.mode = mode
#       if self.mode != "test":
#       # use part of training for validation
#         np.random.seed(1)
#         inds = np.random.permutation(list(range(0, self.dataset_len)))

#         K = 500
#         #self.mode = 'train'
#         if self.mode == "train":
#           self.dataset_len = self.dataset_len - K
#           self.inds = inds[:self.dataset_len]
#         elif self.mode == "val":
#           self.inds = inds[self.dataset_len - K:]
#           self.dataset_len = K

#     def setup(self, name, label, neighbors):
#       config = {}
#       config['loss'] = "cross_entropy"
#       #import pdb; pdb.set_trace()
#       # adjust output size
#       if label == 'age':
#         config['num_output'] = 15
#         config['ignore_label'] = [0, 1]
#       elif label == 'function':
#         config['num_output'] = 208
#         config['ignore_label'] = [0, 1]
#       elif label == 'landuse':
#         config['num_output'] = 13
#         config['ignore_label'] = [1]
#       elif label == 'landcover':
#         config['num_output'] = 9
#         config['ignore_label'] = [0]
#       elif label == 'height':
#         config['num_output'] = 2
#         config['loss'] = "uncertainty"
#       else:
#         raise ValueError('Unknown label.')

#       # setup neighbors
#       config['near_size'] = neighbors

#       return config

#     def open_hdf5(self):
#       self.h5_file = h5py.File(
#         "{}{}/{}".format(self.base_dir, self.name, self.h5_name), "r")

#     def open_streetview(self):
#       fname = 'panos_256*1024_new.h5'   #"panos_calibrated_small.h5"
#       fname_seg = 'seg_256*1024.h5'
#       fname_sat_aligned = 'sat_aligned.h5'
#       #import pdb; pdb.set_trace()
#       self.sv_file = h5py.File("{}{}".format(self.streetview_dir, fname), "r")
#       self.sv_file_seg = h5py.File("{}{}".format(self.streetview_dir, fname_seg), "r")
#       self.sv_file_sat_aligned = h5py.File("{}{}".format(self.aerial_dir, fname_sat_aligned), "r")
#     def open_streetview_seg(self):
#       fname = 'seg_256*1024.h5'
#       self.sv_file_seg = h5py.File("{}{}".format(self.streetview_dir, fname), "r")

#     def __getitem__(self,idx):
#       """Return a data point and its metadata information.

#         Parameters:
#             index - - a random integer for data indexing

#         Returns a dictionary that contains A, B, A_paths and B_paths
#             A (tensor) - - an image in the input domain
#             B (tensor) - - its corresponding image in the target domain
#             A_paths (str) - - image paths
#             B_paths (str) - - image paths (same as A_paths)
#         """
      
#       # import pdb; pdb.set_trace()
#       if not hasattr(self, 'h5_file'):
#         self.open_hdf5()
        
#       if not hasattr(self, 'sv_file'):
#         self.open_streetview()
#         #self.open_streetview_seg()
        
#       if self.mode != "test":
#         idx = self.inds[idx]
#       #import pdb; pdb.set_trace()
#       fname = self.h5_file['fname'][idx]     
#       bbox = self.h5_file['bbox'][idx]
#       label = self.h5_file['label'][idx]
#       near_inds = self.h5_file['near_inds'][idx].astype(int)

#       # from matlab to python indexing
#       near_inds = near_inds - 1

#       # setup neighbors
#       if 0 < self.config['near_size'] <= near_inds.shape[-1]:  # 20 closest street-level panoramas
#         near_inds = near_inds[:self.config['near_size']]
#       else:
#         raise ValueError('Invalid neighbor size.')

#       # near locs, near feats
#       sort_index = np.argsort(near_inds)            #搜索对应的最近的index
#       unsort_index = np.argsort(sort_index)
#       near_locs = self.h5_file['locs'][near_inds[sort_index], ...][unsort_index,
#                                                                   ...]

#       # decode and preprocess panoramas
#       near_streetview = self.sv_file['images'][near_inds[sort_index],
#                                               ...][unsort_index, ...]
#       #near_streetview1 = near_streetview.astype(float)
#       near_streetview_seg = self.sv_file_seg['images'][near_inds[sort_index],
#                                               ...][unsort_index, ...]
#       # add aligned satellite
#       sat_aligned = self.sv_file_sat_aligned['images'][near_inds[sort_index],
#                                               ...][unsort_index, ...]

#       tmp = []
#       for item in near_streetview:
#         tmp_im = preprocess(imageio.imread(io.BytesIO(item))).transpose(
#             2, 0, 1)
#         tmp_im_t = torch.from_numpy(tmp_im).float()
#         tmp_im_t_norm = TF.normalize(tmp_im_t,
#                                     mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#         tmp.append(tmp_im_t_norm)
#       near_streetview = torch.stack(tmp, dim=0)
      
#       tmp_seg = []
#       for item in near_streetview_seg:
#         tmp_im_seg = preprocess(imageio.imread(io.BytesIO(item))).transpose(
#             2, 0, 1)
#         tmp_im_t_seg = torch.from_numpy(tmp_im_seg).float()
#         tmp_im_t_norm_seg = TF.normalize(tmp_im_t_seg,
#                                     mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#         tmp_seg.append(tmp_im_t_norm_seg)
#       near_streetview_seg = torch.stack(tmp_seg, dim=0)
#       #11.14 add
#       tmp_sat_aligned = []
#       for item in sat_aligned:
#         tmp_im_sat_aligned = preprocess(imageio.imread(io.BytesIO(item))).transpose(
#             2, 0, 1)
#         tmp_im_t_sat_aligned = torch.from_numpy(tmp_im_sat_aligned).float()
#         tmp_im_t_norm_sat_aligned = TF.normalize(tmp_im_t_sat_aligned,
#                                     mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#         tmp_sat_aligned.append(tmp_im_t_norm_sat_aligned)
#       sat_aligned = torch.stack(tmp_sat_aligned, dim=0)

#       # form absolute paths
#       fname_image = "{}{}".format(self.aerial_dir, fname.decode())
#       fname_label = "{}{}".format(self.label_dir, label.decode())
#       fname_pano_seg = "{}{}".format(self.streetview_seg_dir, fname.decode().replace('19/',''))
#       image = preprocess(imread(fname_image))  # array
      
#       if self.label == "height":
#         fname_label = "{}.npz".format(fname_label[:-4])
#         label = sparse.load_npz(fname_label).todense()
#         label = label * (1200 / 3937)  # 1 ft (US survey) = 1200/3937 m
#         label = label - label.min()
#       else:
#         label = imread(fname_label)
  

#       t_image = TF.to_tensor(image).float()
#       t_label = torch.from_numpy(label).float()
#       t_bbox = torch.from_numpy(bbox).float()
#       t_near_locs = torch.from_numpy(near_locs).float()
#       t_near_images = near_streetview
#       #import pdb; pdb.set_trace()
      
#       t_pano_seg = near_streetview_seg
#       # paste all 4 satellite images together, transfer shape [256,256] to [256,1024]
      
#       for i in range(1, 4):
#             #t_image = torch.cat((t_image, transforms.ToTensor()(image).float()),2)
#             #image= np.ascontiguousarray(image)
#             #rot_image= image.transpose(1,0,2)[::-1]
            
#             rot_image = image #np.rot90(image,i)
#             rot_image = np.ascontiguousarray(rot_image)
#             t_image = torch.cat((t_image, transforms.ToTensor()(rot_image).float()),2)

#       source_image = t_near_images[1]
      
#       target_image = t_near_images[0]
#       target_loc = t_near_locs[0]
#       source_loc = t_near_locs[1]
#       #source_loc1 = t_near_locs[2]
#       '''
#       disp_vec = np.asarray([target_loc[0]-source_loc[0], target_loc[1]-source_loc[1]])
#       vec = make_gaussian_vector(disp_vec[0], disp_vec[1])
      
#       ###################
#       #if cfg.data_align:
#       theta_x = (180.0 / np.pi) * np.arctan2(disp_vec[1], disp_vec[0])  # first y and then x i.e. arctand (y/x)

#       # angle from y-axis or north
#       theta_y = 90 + theta_x

#       if theta_y < 0:  # fixing negative
#           theta_y += 360

#       column_shift = np.int(
#           theta_y * (cfg.data.image_size[1]/360.0) )   

#       source_image = torch.roll(source_image, column_shift, dims=2)  # rotate columns
#       target_image = torch.roll(target_image, column_shift, dims=2)  # rotate columns
      
#       #################
#       # source_image = AddBorder_tensor(source_image, cfg.data.border_size) # border_size = 0 may led to fault
#       # target_image = AddBorder_tensor(target_image, cfg.data.border_size)
#       '''
#       source_image1 = source_image
#       return {'A':t_near_images[0], 'B':t_image } # A是target, B是condition
  
#     def __len__(self):
#       return self.dataset_len
  
  

# 11.14 aligned
class brooklynqueensdataset():

    def __init__(self,mode):
    #   BaseDataset.__init__(self, opt)
      self.dirA = os.path.join('/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens','brooklyn/overhead') # phase 另算
      self.dirB = os.path.join('/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens','brooklyn/streetview/images')
      
      #self.dir_seg = os.path.join()
      name = 'brooklyn-fc8_landuse'
      neighbors=20
      #import pdb; pdb.set_trace()
      if any(x in name for x in ['brooklyn', 'queens']):
      
        name, label = name.split('_')    # brooklyn-fc8  landcover
        local_name = name.split('-')[0]
        #import pdb; pdb.set_trace()
        data_dir = '/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens/' #"/u/eag-d1/data/near-remote/"
        self.aerial_dir = "{}{}/overhead/".format(data_dir, local_name)  #"{}{}/aerial/".format(data_dir, local_name)
        self.label_dir = "{}{}/labels/{}/".format(data_dir, local_name, label)
        self.streetview_dir = "{}{}/streetview/".format(data_dir, local_name)
        # self.streetview_seg_dir = "{}{}/streetview/seg_fast/".format(data_dir, local_name)
      else:
        raise ValueError('Unknown dataset.')
      
      self.name = name
      self.label = label
      self.base_dir = "/home/x.zhexiao/PyTorch-GAN/data/brooklyn_queens/"  #"/u/eag-d1/scratch/scott/learn/near-remote/data/"
      self.config = self.setup(name, label, neighbors)
      
      #import pdb; pdb.set_trace()
      self.h5_name = "{}_train.h5".format(name) if mode in [
          "train", "val"
      ] else "{}_test.h5".format(name)

      tmp_h5 = h5py.File("{}{}/{}".format(self.base_dir, self.name, self.h5_name),
                        'r')
      self.dataset_len = len(tmp_h5['fname'])
      self.mode = mode
      if self.mode != "test":
      # use part of training for validation
        np.random.seed(1)
        inds = np.random.permutation(list(range(0, self.dataset_len)))

        K = 500
        #self.mode = 'train'
        if self.mode == "train":
          self.dataset_len = self.dataset_len - K
          self.inds = inds[:self.dataset_len]
        elif self.mode == "val":
          self.inds = inds[self.dataset_len - K:]
          self.dataset_len = K

    def setup(self, name, label, neighbors):
      config = {}
      config['loss'] = "cross_entropy"
      #import pdb; pdb.set_trace()
      # adjust output size
      if label == 'age':
        config['num_output'] = 15
        config['ignore_label'] = [0, 1]
      elif label == 'function':
        config['num_output'] = 208
        config['ignore_label'] = [0, 1]
      elif label == 'landuse':
        config['num_output'] = 13
        config['ignore_label'] = [1]
      elif label == 'landcover':
        config['num_output'] = 9
        config['ignore_label'] = [0]
      elif label == 'height':
        config['num_output'] = 2
        config['loss'] = "uncertainty"
      else:
        raise ValueError('Unknown label.')

      # setup neighbors
      config['near_size'] = neighbors

      return config

    def open_hdf5(self):
      self.h5_file = h5py.File(
        "{}{}/{}".format(self.base_dir, self.name, self.h5_name), "r")

    def open_streetview(self):
      fname = 'panos_256*1024_new.h5'   #"panos_calibrated_small.h5"
      fname_seg = 'seg_256*1024.h5'
      fname_sat_aligned = 'sat_aligned.h5'
      #import pdb; pdb.set_trace()
      self.sv_file = h5py.File("{}{}".format(self.streetview_dir, fname), "r")
      self.sv_file_seg = h5py.File("{}{}".format(self.streetview_dir, fname_seg), "r")
      self.sv_file_sat_aligned = h5py.File("{}{}".format(self.aerial_dir, fname_sat_aligned), "r")
    def open_streetview_seg(self):
      fname = 'seg_256*1024.h5'
      self.sv_file_seg = h5py.File("{}{}".format(self.streetview_dir, fname), "r")

    def __getitem__(self,idx):
      """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
      
      # import pdb; pdb.set_trace()
      if not hasattr(self, 'h5_file'):
        self.open_hdf5()
        
      if not hasattr(self, 'sv_file'):
        self.open_streetview()
        #self.open_streetview_seg()
        
      if self.mode != "test":
        idx = self.inds[idx]
      #import pdb; pdb.set_trace()
      fname = self.h5_file['fname'][idx]     
      bbox = self.h5_file['bbox'][idx]
      label = self.h5_file['label'][idx]
      near_inds = self.h5_file['near_inds'][idx].astype(int)

      # from matlab to python indexing
      near_inds = near_inds - 1

      # setup neighbors
      if 0 < self.config['near_size'] <= near_inds.shape[-1]:  # 20 closest street-level panoramas
        near_inds = near_inds[:self.config['near_size']]
      else:
        raise ValueError('Invalid neighbor size.')

      # near locs, near feats
      sort_index = np.argsort(near_inds)            #搜索对应的最近的index
      unsort_index = np.argsort(sort_index)
      near_locs = self.h5_file['locs'][near_inds[sort_index], ...][unsort_index,
                                                                  ...]

      # decode and preprocess panoramas
      near_streetview = self.sv_file['images'][near_inds[sort_index],
                                              ...][unsort_index, ...]
      #near_streetview1 = near_streetview.astype(float)
      near_streetview_seg = self.sv_file_seg['images'][near_inds[sort_index],
                                              ...][unsort_index, ...]
      # add aligned satellite
      sat_aligned = self.sv_file_sat_aligned['images'][near_inds[sort_index],
                                              ...][unsort_index, ...]

      tmp = []
      for item in near_streetview:
        tmp_im = preprocess(imageio.imread(io.BytesIO(item))).transpose(
            2, 0, 1)
        tmp_im_t = torch.from_numpy(tmp_im).float()
        tmp_im_t_norm = TF.normalize(tmp_im_t,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        tmp.append(tmp_im_t_norm)
      near_streetview = torch.stack(tmp, dim=0)
      
      tmp_seg = []
      for item in near_streetview_seg:
        tmp_im_seg = preprocess(imageio.imread(io.BytesIO(item))).transpose(
            2, 0, 1)
        tmp_im_t_seg = torch.from_numpy(tmp_im_seg).float()
        tmp_im_t_norm_seg = TF.normalize(tmp_im_t_seg,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        tmp_seg.append(tmp_im_t_norm_seg)
      near_streetview_seg = torch.stack(tmp_seg, dim=0)
      #11.14 add
      tmp_sat_aligned = []
      for item in sat_aligned:
        tmp_im_sat_aligned = preprocess(imageio.imread(io.BytesIO(item))).transpose(
            2, 0, 1)
        tmp_im_t_sat_aligned = torch.from_numpy(tmp_im_sat_aligned).float()
        tmp_im_t_norm_sat_aligned = TF.normalize(tmp_im_t_sat_aligned,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        tmp_sat_aligned.append(tmp_im_t_norm_sat_aligned)
      sat_aligned = torch.stack(tmp_sat_aligned, dim=0)

      # form absolute paths
      fname_image = "{}{}".format(self.aerial_dir, fname.decode())
      fname_label = "{}{}".format(self.label_dir, label.decode())
      # fname_pano_seg = "{}{}".format(self.streetview_seg_dir, fname.decode().replace('19/',''))
      # image = preprocess(imread(fname_image))  # array
      
      if self.label == "height":
        fname_label = "{}.npz".format(fname_label[:-4])
        label = sparse.load_npz(fname_label).todense()
        label = label * (1200 / 3937)  # 1 ft (US survey) = 1200/3937 m
        label = label - label.min()
      else:
        label = imread(fname_label)
  

      t_image = sat_aligned[0]  #最临近的panorama
      t_label = torch.from_numpy(label).float()
      t_bbox = torch.from_numpy(bbox).float()
      t_near_locs = torch.from_numpy(near_locs).float()
      t_near_images = near_streetview
      #import pdb; pdb.set_trace()
      
      t_pano_seg = near_streetview_seg
      t_tensors = [t_image,t_image,t_image,t_image]
      t_image = torch.cat(t_tensors,2)
      # paste all 4 satellite images together, transfer shape [256,256] to [256,1024]
      
      # for i in range(1, 4):
      #       #t_image = torch.cat((t_image, transforms.ToTensor()(image).float()),2)
      #       #image= np.ascontiguousarray(image)
      #       #rot_image= image.transpose(1,0,2)[::-1]
            
      #       rot_image = image #np.rot90(image,i)
      #       rot_image = np.ascontiguousarray(rot_image)
      #       t_image = torch.cat((t_image, transforms.ToTensor()(rot_image).float()),2)

      source_image = t_near_images[1]
      
      target_image = t_near_images[0]
      target_loc = t_near_locs[0]
      source_loc = t_near_locs[1]
      #source_loc1 = t_near_locs[2]
      '''
      disp_vec = np.asarray([target_loc[0]-source_loc[0], target_loc[1]-source_loc[1]])
      vec = make_gaussian_vector(disp_vec[0], disp_vec[1])
      
      ###################
      #if cfg.data_align:
      theta_x = (180.0 / np.pi) * np.arctan2(disp_vec[1], disp_vec[0])  # first y and then x i.e. arctand (y/x)

      # angle from y-axis or north
      theta_y = 90 + theta_x

      if theta_y < 0:  # fixing negative
          theta_y += 360

      column_shift = np.int(
          theta_y * (cfg.data.image_size[1]/360.0) )   

      source_image = torch.roll(source_image, column_shift, dims=2)  # rotate columns
      target_image = torch.roll(target_image, column_shift, dims=2)  # rotate columns
      
      #################
      # source_image = AddBorder_tensor(source_image, cfg.data.border_size) # border_size = 0 may led to fault
      # target_image = AddBorder_tensor(target_image, cfg.data.border_size)
      '''
      source_image1 = source_image
      return {'A':t_near_images[0], 'B':t_image } # A是target, B是condition
  
    def __len__(self):
      return self.dataset_len
  
