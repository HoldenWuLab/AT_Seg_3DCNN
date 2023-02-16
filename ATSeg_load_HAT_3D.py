import numpy as np
import h5py
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from skimage.transform import resize

class LiverFatDataset(Dataset):
    
    def __init__(self, dataset_dir, h5filenames, subset, transform=None):
        """
        Args:
            dataset_dir (string) : Path to the hdf5 dataset
            h5filenames: The name of hdf5 file (single file or list)
            subset: subset to load: train_, val_, or ev_
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.dir = dataset_dir
        self.name = h5filenames
        self.subset = subset
        self.transform = transform
    
    # Inherited function from pytorch dataset
    def __len__(self):
        [group_data, image_list] = self.Get_h5_group()
        return len(image_list)
    
        
    def __getitem__(self, idx):
        
        # Open and load hdf5 file
        [group_data, image_list] = self.Get_h5_group()
        image_eco0 = self.Get_h5_dataset(group_data, image_list[idx], "Eco0")
        image_eco1 = self.Get_h5_dataset(group_data, image_list[idx], "Eco1")
        water_image = self.Get_h5_dataset(group_data, image_list[idx], "W")
        fat_image = self.Get_h5_dataset(group_data, image_list[idx], "F")
        SAT = self.Get_h5_dataset(group_data, image_list[idx], "SAT")
        VAT = self.Get_h5_dataset(group_data, image_list[idx], "VAT")
        Subject_ID = self.Get_h5_dataset(group_data, image_list[idx], "Subject_ID")
        
        # IMPORTANT: the below variables should be in float format, or it creates problem in the resize function
        image_eco0 = image_eco0.astype('float')
        image_eco1 = image_eco1.astype('float')
        water_image = water_image.astype('float')
        fat_image = fat_image.astype('float')
        SAT = SAT.astype('float')
        VAT = VAT.astype('float')
        
        
        # Resize all the array to the same number of slices
        Nsl = 64
        mat_size = image_eco0.shape[1]
        if image_eco0.shape[1] is not mat_size or image_eco0.shape[2] is not mat_size:
            image_eco0 = resize(image_eco0, (Nsl, mat_size, mat_size), anti_aliasing=True, preserve_range = True)
            image_eco1 = resize(image_eco1, (Nsl, mat_size, mat_size), anti_aliasing=True, preserve_range = True)
            fat_image = resize(fat_image, (Nsl, mat_size, mat_size), anti_aliasing=True, preserve_range = True)
            water_image = resize(water_image, (Nsl, mat_size, mat_size), anti_aliasing=True, preserve_range = True) 
            SAT = np.squeeze(np.round(resize(SAT, (Nsl, mat_size, mat_size), anti_aliasing=True, preserve_range = True)))
            VAT = np.squeeze(np.round(resize(VAT, (Nsl, mat_size, mat_size), anti_aliasing=True, preserve_range = True)))
            
        if image_eco0.shape[2] is not Nsl:
            image_eco0 = resize(image_eco0, (Nsl, mat_size, mat_size), anti_aliasing=True, preserve_range = True)
            image_eco1 = resize(image_eco1, (Nsl, mat_size, mat_size), anti_aliasing=True, preserve_range = True)
            fat_image = resize(fat_image, (Nsl, mat_size, mat_size), anti_aliasing=True, preserve_range = True)
            water_image = resize(water_image, (Nsl, mat_size, mat_size), anti_aliasing=True, preserve_range = True) 
            SAT = np.squeeze(np.round(resize(SAT, (Nsl, mat_size, mat_size), anti_aliasing=True, preserve_range = True)))
            VAT = np.squeeze(np.round(resize(VAT, (Nsl, mat_size, mat_size), anti_aliasing=True, preserve_range = True)))
        BG = np.ones_like(SAT)-SAT-VAT

        image_eco0_255 = ((image_eco0 - np.amin(image_eco0)) * (1/(np.amax(image_eco0) - np.amin(image_eco0)) * 255))   
        image_eco1_255 = ((image_eco1 - np.amin(image_eco1)) * (1/(np.amax(image_eco1) - np.amin(image_eco1)) * 255))
        fat_image_255 = ((fat_image - np.amin(fat_image)) * (1/(np.amax(fat_image) - np.amin(fat_image)) * 255))
        water_image_255 = ((water_image- np.amin(water_image)) * (1/(np.amax(water_image) - np.amin(water_image)) * 255))
        mask = np.stack((VAT, SAT, BG), 3)

        # For inputs of echo0, water and fat (TE_{OP}+W+F)
        input_image = np.squeeze(np.stack((water_image_255, fat_image_255, image_eco0_255), 3))

        
        # Build sample of the dataset and transformed into torch dataset
        sample = {'image_Eco0': image_eco0_255, 'image_Eco1': image_eco1_255, 'fat_image': fat_image_255, 'water_image': water_image_255, 'input_image': input_image, 'mask': mask, 'Subject_ID': Subject_ID}
        if self.transform:
            sample = self.transform(sample)
        return sample
               
    # data load from hdf5 file
    def Get_h5_group(self):
        """
           Load group from h5 dataset
           List subgroup from the group
        """ 
        # this solves the problem of being unable to create/open the .h5 file
        f = h5py.File(self.dir+self.name, "r")
        
        try:
            group_key = list(f.keys())
            group_data = f.get(self.subset)
            subgroup_key = list(group_data.keys())
        except:
            print('error in geth5group')
            print ("No such group, select the group from: "  + "\"" + '", "'.join(group_key) + "\"")
            raise
        f.close
        return group_data, subgroup_key
    
    def Get_h5_dataset(self, group_data, subgroup_name, dataset_name, debug=False):
        """
           Load dataset from group/subgroup
           1) subgroup_idx: index of image subgroup to load
           2) dataset_name: name of dataset to load
        """

        # load image subgroup from group
        subgroup_key = list(group_data.keys())
        subgroup_data = group_data.get(subgroup_name)
        try:
            # load dataset from image subgroup
            dataset_key = list(subgroup_data.keys())
            dataset = subgroup_data.get(dataset_name)[...]
        except Exception as e:
            print (f"Exception={e}")
            print ("No such dataset, select the dataset name from: "  + "\"" + '", "'.join(dataset_key) + "\"")
            raise
        return dataset
    
    
def load_LiverFatDataset(train_batch_size, test_batch_size, debug_load=False):

    # Please remember to update with your own directories!
    liverfat_h5_path = "/skafali/AT Seg/training/h5/"
    liverfat_h5_name = "ATSeg_HAT_training_3D.h5"
    liverfat_transformed_train_dataset = LiverFatDataset(liverfat_h5_path, liverfat_h5_name, "Train_image",\
                                                   transform = transforms.Compose([ToTensor()]))
    print(liverfat_h5_name + " contains " + str(len(liverfat_transformed_train_dataset)) + " training subjects with " + 
         str(len(liverfat_transformed_train_dataset)*len(liverfat_transformed_train_dataset[0]["image_Eco0"])) + " images")

    # Please remember to update with your own directories!
    liverfat_h5_path = "/skafali/AT Seg/validation/h5/"
    liverfat_h5_name = "ATSeg_HAT_BH_validation_3D.h5"
    liverfat_transformed_validation_dataset = LiverFatDataset(liverfat_h5_path, liverfat_h5_name, "Validations_image",\
                                                   transform = transforms.Compose([ToTensor()]))
    print(liverfat_h5_name + " contains " + str(len(liverfat_transformed_validation_dataset)) + " validation subjects with " + 
         str(len(liverfat_transformed_validation_dataset)*len(liverfat_transformed_validation_dataset[0]["image_Eco0"])) + " images")

    # Please remember to update with your own directories!
    liverfat_h5_path = "/skafali/AT Seg/test/h5/"
    liverfat_h5_name = "ATSeg_HAT_BH_test_3D.h5"
    liverfat_transformed_test_dataset = LiverFatDataset(liverfat_h5_path, liverfat_h5_name, "Test_image",\
                                                   transform = transforms.Compose([ToTensor()]))
    print(liverfat_h5_name + " contains " + str(len(liverfat_transformed_test_dataset)) + " test subjects with " + 
         str(len(liverfat_transformed_test_dataset)*len(liverfat_transformed_test_dataset[0]["image_Eco0"])) + " images") 
    

    # take the whole data and divide them into batches and return
    liverfat_train_loader, liverfat_validation_loader, liverfat_test_loader = load_batches(liverfat_transformed_train_dataset, liverfat_transformed_validation_dataset, liverfat_transformed_test_dataset, train_batch_size,test_batch_size, debug_load)
    
    return liverfat_transformed_train_dataset, liverfat_transformed_validation_dataset, liverfat_transformed_test_dataset, liverfat_train_loader, liverfat_validation_loader, liverfat_test_loader


def get_ax(rows=1, cols=1, size=8):

    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))

    return ax


def load_batches(liverfat_transformed_train_dataset, liverfat_transformed_validation_dataset, liverfat_transformed_test_dataset, train_batch_size,test_batch_size, debug_load=False):


    liverfat_train_loader = DataLoader(liverfat_transformed_train_dataset,
                              train_batch_size,
                              shuffle=True, num_workers=1)

    liverfat_validation_loader = DataLoader(liverfat_transformed_validation_dataset,
                              test_batch_size,
                              shuffle=False, num_workers=1)

    liverfat_test_loader = DataLoader(liverfat_transformed_test_dataset,
                              test_batch_size,
                              shuffle=False, num_workers=1)

    if debug_load:
        for i_batch, sample_batched in enumerate(liverfat_train_loader):

             print(i_batch, 
                  sample_batched['image'].size(),
                  sample_batched['mask'].size())

        for i_batch, sample_batched in enumerate(liverfat_validation_loader):

            print(i_batch,
              sample_batched['image'].size(),
              sample_batched['mask'].size())

        for i_batch, sample_batched in enumerate(liverfat_test_loader):

            print(i_batch,
              sample_batched['image'].size(),
              sample_batched['mask'].size())

    return liverfat_train_loader, liverfat_validation_loader, liverfat_test_loader

   
class ToTensor(object):
    # Convert ndarrays in sample to Tensors

    def __call__(self, sample):    
        image_eco0, image_eco1, fat_image, water_image, image, mask, Subject_ID  = sample['image_Eco0'], sample['image_Eco1'], sample['fat_image'], sample['water_image'], sample['input_image'], sample['mask'], sample['Subject_ID']
        image = image.transpose((3,0,1,2)).astype('float')
        mask = mask.transpose((3,0,1,2)).astype('float')
        return {'image_Eco0': torch.from_numpy(image_eco0),
                'image_Eco1': torch.from_numpy(image_eco1),
                'fat_image': torch.from_numpy(fat_image),
                'water_image': torch.from_numpy(water_image),
                'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask),
                'Subject_ID': Subject_ID}
    

