import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.functional as f
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from skimage.transform import resize
import sklearn.metrics
import scipy
from statsmodels.graphics.agreement import mean_diff_plot
from losses import LogWeightedDICELossMultiClass3D
from models_all import BACompDenseUNet3D
import torch.optim as optim
import os
from tqdm import tqdm
from torch.autograd import Variable
import statsmodels.api as sm
from ATSeg_load_HAT_3D import LiverFatDataset
from ATSeg_load_HAT_3D import load_LiverFatDataset
from ATSeg_load_HAT_3D import ToTensor
import time
from ICC import icc
import pickle
import datetime
import time
import GPUtil
from MedicalEvaluationMetrics import sensitivity
from MedicalEvaluationMetrics import specificity
from MedicalEvaluationMetrics import DICECoeff
from MedicalEvaluationMetrics import surfd
from find_ID_type import find_ID, find_image_type, find_slice

# select the free GPU automatically
gpu_id = GPUtil.getFirstAvailable(order = 'first', maxLoad=0.05, maxMemory=0.05, attempts=600, interval=60, verbose=False)[0]
print(f'The GPU ID is: {gpu_id}')
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) 

"""max_allowed_memory_alloc = 2500
free_gpu_found = False
while not free_gpu_found:
    for gpu in range(10):
        try:
            memory_allocated = torch.cuda.memory_allocated(gpu)
            if memory_allocated < max_allowed_memory_alloc:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
                free_gpu_found = True
                break
        except Exception as e:
            continue
    print(f"{datetime.now().replace(microsecond=0)}: all GPUs are in use: will try again in 5 minutes")
    time.sleep(300)
"""

train_batch_size = 1
test_batch_size = 1
use_gpu = True
lr = 0.0005
num_epochs = 25
loss_threshold = 0.5 #0.3 #0.4 #0.5
loss_list = np.zeros((num_epochs))
val_loss_list = np.zeros((num_epochs))
save_vat = 'VATMasks'
save_sat = 'SATMasks'
save_bg = 'BGMasks'
save = 'ALLMasks'

test_accuracy = False
train_accuracy = False

# Set up model, optimizer and loss
model = BACompDenseUNet3D(3,3)
# model = nn.parallel.DataParallel(model, device_ids= [0,1])
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = LogWeightedDICELossMultiClass3D() 
t = torch.cuda.get_device_properties(0).total_memory
main_dir_out_files = '/radraid/skafali/AT Seg HAT/AT_seg/Codes/Adipose_Tissue_Segmentation_8_4_2022_HAT/out-files/'
main_filename = 'BA-CompDenseNet-3D-HAT-FBDL'

def train(epoch):
    
    # SGK: random shuffling of the batches are incorporated into the code right now. 
    model.train()
    max_batch_num = len(liverfat_transformed_train_dataset)
    print(f"max_batch_num is: {max_batch_num}")
    Nch, Nsl, imgN, imgN  = liverfat_train_loader.dataset[0]['image'].shape
    num_classes, Nsl, imgN, imgN = liverfat_train_loader.dataset[0]['mask'].shape
    temp_img = torch.cuda.FloatTensor(train_batch_size, Nch, Nsl, imgN,imgN)
    temp_mask = torch.cuda.FloatTensor(train_batch_size, num_classes, Nsl, imgN,imgN)
    batch_data = {'image': temp_img, 'mask': temp_mask}
        
    for batch_number in range(max_batch_num):    
        # random sampling of the batches
        batch_idxs = np.random.choice(np.arange(max_batch_num), train_batch_size, replace=True)
        print("random batch idx's are: " + str(batch_idxs))
        
        # taking samples according to the batch size
        for batch_idx in np.arange(train_batch_size):
            temp_batch_data = liverfat_train_loader.dataset[batch_idxs[batch_idx]]
            batch_data['image'][batch_idx,:,:,:,:] = temp_batch_data['image']
            batch_data['mask'][batch_idx,:,:,:,:] = temp_batch_data['mask']
            
        # constructing the images and masks according to the batch size
        image = batch_data['image']
        mask = batch_data['mask']
             
        if use_gpu:
            image, mask = image.cuda(), mask.cuda()

        image = image.float()
        mask = mask.float()
        optimizer.zero_grad()
        
#         r0 = torch.cuda.memory_reserved(0) 
        a0 = torch.cuda.memory_allocated(0)
        f0 = t-a0  # free inside reserved
#         print(f"free GPU memory before emptying of cache: {f0}")
        torch.cuda.empty_cache()
        
        output = model(image)
        loss = criterion(output, mask, loss_threshold)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage DICE Loss: {:.6f}'.format(
            epoch, batch_number * len(image), str(max_batch_num),
            100. * batch_number / max_batch_num, loss.item()))
        
#         r_end = torch.cuda.memory_reserved(0) 
        a_end = torch.cuda.memory_allocated(0)
        f_end = t-a_end  # free inside reserved
#         print(f"free GPU memory after emptying of cache: {f_end}")

    return output, mask, loss

def test(liverfat_train_loader, liverfat_validation_loader, liverfat_test_loader, train_accuracy=False, test_accuracy=False):
    test_loss = 0

    if train_accuracy:
        print("Training Accuracy")
        loader = liverfat_train_loader
    elif test_accuracy:
        print("Test Accuracy")
        loader = liverfat_test_loader
    else: 
        print("Validation Accuracy")
        loader = liverfat_validation_loader
        
    for batch_idx in range(len(loader)):
        batch_data = loader.dataset[batch_idx]
        image = batch_data['image']
        mask = batch_data['mask']
        
        image = image[np.newaxis,:,:,:]
        mask = mask[np.newaxis,:,:,:]
        
        if use_gpu:
            image, mask = image.cuda(), mask.cuda()

        image = image.float()
        mask = mask.float()
        output = model(image)
        maxes, out = torch.max(output, 1, keepdim=True)

        # output SAT, VAT and BG masks
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-outs.npy'.format(save_vat,
                                                                            batch_idx),
                                                                            output[:,0,:,:,:].data.cpu().numpy())
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-outs.npy'.format(save_sat,
                                                                            batch_idx),
                                                                            output[:,1,:,:,:].data.cpu().numpy())
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-outs.npy'.format(save_bg,
                                                                            batch_idx),
                                                                            output[:,2,:,:,:].data.cpu().numpy())
        # ground truth masks
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-masks.npy'.format(save_vat,
                                                                            batch_idx),
                                                                            mask[:,0,:,:,:].data.byte().cpu().numpy())
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-masks.npy'.format(save_sat,
                                                                            batch_idx),
                                                                            mask[:,1,:,:,:].data.byte().cpu().numpy())
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-masks.npy'.format(save_bg,
                                                                            batch_idx),
                                                                            mask[:,2,:,:,:].data.byte().cpu().numpy())

        # input images
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-images.npy'.format(save,
                                                                              batch_idx),
                                                                              image.data.float().cpu().numpy())

        test_loss += criterion(output, mask, loss_threshold).item()

    # Average Dice Coefficient
    test_loss /= len(loader)
    if train_accuracy:
        print('\nTraining Set: Average DICE Coefficient: {:.4f})\n'.format(
            test_loss))
    else:
        print('\nTest Set: Average DICE Loss: {:.4f})\n'.format(
            test_loss))
        
        
# calculate the number of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_param = count_parameters(model)
print("Number of parameters for 3D BA Comp DenseUNet is: " + str(num_param))

# Start Training
ii = 0
for ep in tqdm(range(num_epochs)):
    t0 = time.time()
    liverfat_transformed_train_dataset, liverfat_transformed_validation_dataset, liverfat_transformed_test_dataset, liverfat_train_loader, liverfat_validation_loader, liverfat_test_loader = load_LiverFatDataset(train_batch_size, test_batch_size)
    output, mask, loss = train(ii)
    t_end = time.time() - t0
    t_end = t_end/60
    print('{} minutes for an epoch'.format(t_end))
    loss_list[ii] = loss.cpu().detach().numpy()
    torch.cuda.empty_cache()
    with torch.no_grad():
        # validation or test loss
        val_loss_list[ii] = test(liverfat_train_loader, liverfat_validation_loader, liverfat_test_loader, train_accuracy=train_accuracy, test_accuracy=test_accuracy)
    ii=ii+1
    if ii % 5 == 0:
        torch.save(model.state_dict(), main_dir_out_files+main_filename+'-{}-{}-{}-{}-{}-{}'.format(train_batch_size,
                                                                                                    num_epochs,
                                                                                                    lr,
                                                                                                    'ALL',
                                                                                                     loss_threshold,
                                                                                                     ii))
    
    

# loading data
if test_accuracy == False:
    max_batch_num_test = len(liverfat_validation_loader)
else: 
    max_batch_num_test = len(liverfat_test_loader)
    
batch_number = 0
threshold = 0.6
Nsl = 64
gt_VAT = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
gt_SAT = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
gt_BG = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
results_VAT = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
results_SAT = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
results_BG = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
fat_image = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
water_image = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
eco0 = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))

while batch_number < max_batch_num_test:
    # VAT
    gt_VAT[batch_number,:,:,:,:] = np.load (main_dir_out_files+main_filename+'-VATMasks-batch-%d-masks.npy' % batch_number)
    results_VAT[batch_number,:,:,:,:] = np.load (main_dir_out_files+main_filename+'-VATMasks-batch-%d-outs.npy'% batch_number)

    # SAT
    gt_SAT[batch_number,:,:,:,:] = np.load (main_dir_out_files+main_filename+'-SATMasks-batch-%d-masks.npy'% batch_number)
    results_SAT[batch_number,:,:,:,:] = np.load (main_dir_out_files+main_filename+'-SATMasks-batch-%d-outs.npy'% batch_number)

    # BG
    gt_BG[batch_number,:,:,:] = np.load (main_dir_out_files+main_filename+'-BGMasks-batch-%d-masks.npy'% batch_number)
    results_BG[batch_number,:,:,:,:] = np.load (main_dir_out_files+main_filename+'-BGMasks-batch-%d-outs.npy'% batch_number)
    
    # echo0, water and fat images
    tmp_image = np.squeeze(np.load (main_dir_out_files+main_filename+'-ALLMasks-batch-%d-images.npy'% batch_number))
    water_image[batch_number,:,:,:,:] = tmp_image[0,:,:,:]
    fat_image[batch_number,:,:,:,:] = tmp_image[1,:,:,:]
    eco0[batch_number,:,:,:,:] = tmp_image[2,:,:,:]
    batch_number = batch_number + 1

# SGK: combine the results
results_all = np.stack((results_VAT, results_SAT, results_BG), 2)
gt_all = np.stack((gt_VAT, gt_SAT, gt_BG), 2)


# loading data
if test_accuracy == False:
    max_batch_num_test = len(liverfat_validation_loader)
else: 
    max_batch_num_test = len(liverfat_test_loader)
    
batch_number = 0
threshold = 0.6
Nsl = 64
gt_VAT = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
gt_SAT = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
gt_BG = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
results_VAT = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
results_SAT = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
results_BG = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
fat_image = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
water_image = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
eco0 = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))
eco1 = np.zeros((max_batch_num_test,test_batch_size,Nsl,192,192))

while batch_number < max_batch_num_test:
    # VAT
    gt_VAT[batch_number,:,:,:,:] = np.load (main_dir_out_files+main_filename+'-VATMasks-batch-%d-masks.npy' % batch_number)
    results_VAT[batch_number,:,:,:,:] = np.load (main_dir_out_files+main_filename+'-VATMasks-batch-%d-outs.npy'% batch_number)

    # SAT
    gt_SAT[batch_number,:,:,:,:] = np.load (main_dir_out_files+main_filename+'-SATMasks-batch-%d-masks.npy'% batch_number)
    results_SAT[batch_number,:,:,:,:] = np.load (main_dir_out_files+main_filename+'-SATMasks-batch-%d-outs.npy'% batch_number)

    # BG
    gt_BG[batch_number,:,:,:] = np.load (main_dir_out_files+main_filename+'-BGMasks-batch-%d-masks.npy'% batch_number)
    results_BG[batch_number,:,:,:,:] = np.load (main_dir_out_files+main_filename+'-BGMasks-batch-%d-outs.npy'% batch_number)
    
    # echo0, water and fat images
    tmp_image = np.squeeze(np.load (main_dir_out_files+main_filename+'-ALLMasks-batch-%d-images.npy'% batch_number))
    eco0[batch_number,:,:,:,:] = tmp_image[0,:,:,:]
    eco1[batch_number,:,:,:,:] = tmp_image[1,:,:,:]

    batch_number = batch_number + 1

# SGK: combine the results
results_all = np.stack((results_VAT, results_SAT, results_BG), 2)
gt_all = np.stack((gt_VAT, gt_SAT, gt_BG), 2)


# Returns index of x in arr if present, else -1
def binary_search(arr, low, high, x):
    # Check base case
    if high >= low:
        mid = (high + low) // 2
        print(f"subject {int(arr[mid]['Subject_ID'])}")
        
        # If element is present at the middle itself
        if int(arr[mid]['Subject_ID']) == x:
            return mid
        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif int(arr[mid]['Subject_ID']) > x:
            return binary_search(arr, low, mid - 1, x)
        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, high, x)
 
    else:
        # Element is not present in the array
        return -1


# Find a specific patient    
def find_patient3D(find_subject_id, dataset, results_SAT, results_VAT, display_on):
    if test_accuracy == False:
        main_dir_fig_files = '/radraid/skafali/AT Seg HAT/AT_seg/Codes/Adipose_Tissue_Segmentation_8_4_2022_HAT/figs/test/'
    else:
        main_dir_fig_files = '/radraid/skafali/AT Seg HAT/AT_seg/Codes/Adipose_Tissue_Segmentation_8_4_2022_HAT/figs/validation/'
    
    idx = binary_search(dataset, 0, len(dataset), int(find_subject_id))
    print(idx)
    print(f'for beginning {datetime.datetime.now()}')
    try:
        d = dataset[idx]
        gt_SAT = d['mask'][1,:,:,:].cpu().detach().numpy()
        gt_VAT = d['mask'][0,:,:,:].cpu().detach().numpy()
        fat_image = d['fat_image'].cpu().detach().numpy()
        water_image = d['water_image'].cpu().detach().numpy()
        eco0 = d['image_Eco0'].cpu().detach().numpy()
        idx_patient = idx
        print(f'for step 1 {datetime.datetime.now()}')
    except TypeError as e:
        print(e)
        raise
    
    # SAT
    res_SAT_patient = np.squeeze(results_SAT[idx_patient,:,:,:])
    gt_SAT_patient = np.squeeze(gt_SAT)
    
    # VAT
    res_VAT_patient = np.squeeze(results_VAT[idx_patient,:,:,:])
    gt_VAT_patient = np.squeeze(gt_VAT)
    
    # fat image
    fat_image_patient = np.squeeze(fat_image)
    
    # water image
    water_image_patient = np.squeeze(water_image)
    
    # Eco0
    eco0_patient = np.squeeze(eco0)

    if display_on: 
        plt.figure()
        ax = plt.subplot(2,4,1)
        ax.imshow(gt_SAT_patient[12,:,:].reshape(192,192), cmap = 'gray')
        ax.set_title('GT SAT segmentation')
        
        ax = plt.subplot(2,4,2)
        ax.imshow(res_SAT_patient[12,:,:].reshape(192,192), cmap = 'gray')
        ax.set_title('GT SAT segmentation')
        
        ax = plt.subplot(2,4,3)
        ax.imshow(gt_VAT_patient[12,:,:].reshape(192,192), cmap = 'gray')
        ax.set_title('GT VAT segmentation')
        
        ax = plt.subplot(2,4,4)
        ax.imshow(res_VAT_patient[12,:,:].reshape(192,192), cmap = 'gray')
        ax.set_title('GT VAT segmentation')

        ax = plt.subplot(2,4,5)
        ax.imshow(fat_image_patient[12,:,:].reshape(192,192), cmap = 'gray')
        ax.set_title('Fat image')
        
        ax = plt.subplot(2,4,6)
        ax.imshow(water_image_patient[12,:,:].reshape(192,192), cmap = 'gray')
        ax.set_title('Water image')
        
        ax = plt.subplot(2,4,7)
        ax.imshow(eco0_patient[12,:,:].reshape(192,192), cmap = 'gray')
        ax.set_title('Eco0 image')
        
        fig = plt.gcf()
        fig.set_size_inches(16.5, 8.5)
        
        fig.savefig(main_dir_fig_files+str(find_subject_id)+'n50.png')
        plt.close(fig)
        
    return gt_SAT_patient, res_SAT_patient, gt_VAT_patient, res_VAT_patient, fat_image_patient, water_image_patient, eco0_patient



# Signal fat fraction (SFF) and Volume measurements for a specific patient 
def patient_measurements(find_subject_id, gt_SAT_patient, res_SAT_patient, gt_VAT_patient, res_VAT_patient, fat_image_patient, water_image_patient, eco0_patient, threshold, num_sl):
    
    SFF_patient = fat_image_patient/(fat_image_patient+water_image_patient)*100
    
    SAT_GT_SFF_tmp = SFF_patient*gt_SAT_patient
    SAT_RES_SFF_tmp = SFF_patient*(res_SAT_patient>threshold)

    VAT_GT_SFF_tmp = SFF_patient*gt_VAT_patient
    VAT_RES_SFF_tmp = SFF_patient*(res_VAT_patient>threshold)
    
    gt_BG_patient = np.ones_like(gt_VAT_patient)
    gt_BG_patient = gt_BG_patient - gt_VAT_patient - gt_SAT_patient

    plt.figure()
    ax = plt.subplot(2,4,1)
    ax.imshow(np.transpose(gt_SAT_patient[30,:,:].reshape(192,192)), cmap = 'gray')
    ax.set_title(str(find_subject_id)+' GT SAT')
    ax = plt.subplot(2,4,2)
    ax.imshow(np.transpose(res_SAT_patient[30,:,:].reshape(192,192)), cmap = 'gray')
    ax.set_title(str(find_subject_id)+' resultant SAT')
    ax = plt.subplot(2,4,3)
    ax.imshow(np.transpose(gt_VAT_patient[30,:,:].reshape(192,192)), cmap = 'gray')
    ax.set_title(str(find_subject_id)+' GT VAT')
    ax = plt.subplot(2,4,4)
    ax.imshow(np.transpose(res_VAT_patient[30,:,:].reshape(192,192)), cmap = 'gray')
    ax.set_title(str(find_subject_id)+' resultant VAT')
    ax = plt.subplot(2,4,5)
    ax.imshow(np.transpose(gt_BG_patient[30,:,:].reshape(192,192)), cmap = 'gray')
    ax.set_title(str(find_subject_id)+' GT Background')
    ax = plt.subplot(2,4,6)
    ax.imshow(np.transpose(fat_image_patient[30,:,:].reshape(192,192)), cmap = 'gray')
    ax.set_title(str(find_subject_id)+' Fat image')
    ax = plt.subplot(2,4,7)
    ax.imshow(np.transpose(water_image_patient[30,:,:].reshape(192,192)), cmap = 'gray')
    ax.set_title(str(find_subject_id)+' Water image')
    ax = plt.subplot(2,4,8)
    ax.imshow(np.transpose(eco0_patient[30,:,:].reshape(192,192)), cmap = 'gray')
    ax.set_title(str(find_subject_id)+' Eco0')


    fig = plt.gcf()
    fig.set_size_inches(16.5, 8.5)

    SAT_GT_SFF_slice = np.mean(np.mean(SAT_GT_SFF_tmp,2),1)
    SAT_RES_SFF_slice = np.mean(np.mean(SAT_RES_SFF_tmp,2),1)
    SAT_GT_SFF = np.mean(SAT_GT_SFF_tmp[SAT_GT_SFF_tmp>0].reshape(-1,1))
    SAT_RES_SFF = np.mean(SAT_RES_SFF_tmp[SAT_RES_SFF_tmp>0].reshape(-1,1))

    VAT_GT_SFF_slice = np.mean(np.mean(VAT_GT_SFF_tmp,2),1)
    VAT_RES_SFF_slice = np.mean(np.mean(VAT_RES_SFF_tmp,2),1)
    VAT_GT_SFF = np.mean(VAT_GT_SFF_tmp[VAT_GT_SFF_tmp>0].reshape(-1,1))
    VAT_RES_SFF = np.mean(VAT_RES_SFF_tmp[VAT_RES_SFF_tmp>0].reshape(-1,1))
    
    SAT_GT_VOL_slice = np.zeros((int(num_sl),1), dtype='float')
    SAT_RES_VOL_slice = np.zeros((int(num_sl),1), dtype='float')
    VAT_GT_VOL_slice = np.zeros((int(num_sl),1), dtype='float')
    VAT_RES_VOL_slice = np.zeros((int(num_sl),1), dtype='float')
    
    SAT_GT_VOL_tmp2 = np.zeros((int(num_sl),1), dtype='float')
    SAT_RES_VOL_tmp2 = np.zeros((int(num_sl),1), dtype='float')
    VAT_GT_VOL_tmp2 = np.zeros((int(num_sl),1), dtype='float')
    VAT_RES_VOL_tmp2 = np.zeros((int(num_sl),1), dtype='float')
    
    SAT_DICE_pat = np.zeros((int(num_sl),1))
    VAT_DICE_pat = np.zeros((int(num_sl),1))
#     SAT_Hausdorf_pat = np.zeros((int(num_sl),1))
#     VAT_Hausdorf_pat = np.zeros((int(num_sl),1))
    
    for sl in range(int(num_sl)):

        SAT_GT_VOL_tmp = (gt_SAT_patient[sl,:,:].reshape(-1,1))
        SAT_RES_VOL_tmp = (res_SAT_patient[sl,:,:].reshape(-1,1))>threshold

        SAT_GT_VOL_tmp2[sl] = np.count_nonzero(SAT_GT_VOL_tmp.reshape(-1,1))
        SAT_RES_VOL_tmp2[sl] = np.count_nonzero(SAT_RES_VOL_tmp.reshape(-1,1))

        VAT_GT_VOL_tmp = (gt_VAT_patient[sl,:,:].reshape(-1,1))
        VAT_RES_VOL_tmp = (res_VAT_patient[sl,:,:].reshape(-1,1))>threshold

        VAT_GT_VOL_tmp2[sl] = np.count_nonzero(VAT_GT_VOL_tmp.reshape(-1,1))
        VAT_RES_VOL_tmp2[sl] = np.count_nonzero(VAT_RES_VOL_tmp.reshape(-1,1))
        
        SAT_DICE_pat[sl] = DICECoeff(res_SAT_patient[sl,:,:]>threshold,gt_SAT_patient[sl,:,:])
        VAT_DICE_pat[sl] = DICECoeff(res_VAT_patient[sl,:,:]>threshold,gt_VAT_patient[sl,:,:])
        
#         SAT_Hausdorf_pat[sl], _, _ = scipy.spatial.distance.directed_hausdorff(res_SAT_patient[sl,:,:]>threshold,gt_SAT_patient[sl,:,:])
#         VAT_Hausdorf_pat[sl], _, _ = scipy.spatial.distance.directed_hausdorff(res_VAT_patient[sl,:,:]>threshold,gt_VAT_patient[sl,:,:])
    
    # 3.5mm slice thickness with 2.08mm 2D iso resolution, then convert back to mm3 to mL
    SAT_GT_VOL = 3.5*2.08*2.08*0.001*np.sum(SAT_GT_VOL_tmp2)
    SAT_RES_VOL = 3.5*2.08*2.08*0.001*np.sum(SAT_RES_VOL_tmp2)

    VAT_GT_VOL = 3.5*2.08*2.08*0.001*np.sum(VAT_GT_VOL_tmp2)
    VAT_RES_VOL = 3.5*2.08*2.08*0.001*np.sum(VAT_RES_VOL_tmp2)
    
    med_SAT_DICE_pat = np.median(SAT_DICE_pat)
    med_VAT_DICE_pat = np.median(VAT_DICE_pat)
    iqr_SAT_DICE_pat = scipy.stats.iqr(SAT_DICE_pat)
    iqr_VAT_DICE_pat = scipy.stats.iqr(VAT_DICE_pat)
    
    mean_SAT_DICE_pat = np.mean(SAT_DICE_pat)
    mean_VAT_DICE_pat = np.mean(VAT_DICE_pat)
    std_SAT_DICE_pat = np.std(SAT_DICE_pat)
    std_VAT_DICE_pat = np.std(VAT_DICE_pat)
    
#     mean_SAT_Hausdorf_pat = np.mean(SAT_Hausdorf_pat)
#     mean_VAT_Hausdorf_pat = np.mean(VAT_Hausdorf_pat)
    
    SAT_surface_distance = surfd(res_SAT_patient>threshold, gt_SAT_patient, [2.08, 2.08, 3.5],1)
    VAT_surface_distance = surfd(res_VAT_patient>threshold, gt_VAT_patient, [2.08, 2.08, 3.5],1)
    
    threeD_SAT_HD = SAT_surface_distance.mean()
    threeD_VAT_HD = VAT_surface_distance.mean()
    
    threeD_SAT_DICE_pat = DICECoeff(res_SAT_patient>threshold,gt_SAT_patient)
    threeD_VAT_DICE_pat = DICECoeff(res_VAT_patient>threshold,gt_VAT_patient)
    
    threeD_SAT_KAP_pat = sklearn.metrics.cohen_kappa_score((res_SAT_patient>threshold).reshape(-1),gt_SAT_patient.reshape(-1))
    threeD_VAT_KAP_pat = sklearn.metrics.cohen_kappa_score((res_VAT_patient>threshold).reshape(-1),gt_VAT_patient.reshape(-1))
        
    SAT_VOL_SIM = 1-np.absolute(np.sum(SAT_GT_VOL_tmp2)-np.sum(SAT_RES_VOL_tmp2))/(np.sum(SAT_GT_VOL_tmp2)+np.sum(SAT_RES_VOL_tmp2))
    VAT_VOL_SIM = 1-np.absolute(np.sum(VAT_GT_VOL_tmp2)-np.sum(VAT_RES_VOL_tmp2))/(np.sum(VAT_GT_VOL_tmp2)+np.sum(VAT_RES_VOL_tmp2))
    
    
    SAT_sens_pat = sensitivity(res_SAT_patient>threshold,gt_SAT_patient)
    SAT_spec_pat = specificity(res_SAT_patient>threshold,gt_SAT_patient)
    
    VAT_sens_pat = sensitivity(res_VAT_patient>threshold,gt_VAT_patient)
    VAT_spec_pat = specificity(res_VAT_patient>threshold,gt_VAT_patient)
    
    diff_SAT_patient1 = gt_SAT_patient - res_SAT_patient
    diff_VAT_patient1 = gt_VAT_patient - res_VAT_patient

    diff_SAT_patient2 = res_SAT_patient - gt_SAT_patient
    diff_VAT_patient2 = res_VAT_patient - gt_VAT_patient

    false_neg = (diff_SAT_patient1 > 0.98) + (diff_VAT_patient1 > 0.98)
    false_pos = (diff_SAT_patient2 > 0.98) + (diff_VAT_patient2 > 0.98)
    
    false_neg_patient = np.sum((false_neg==True).reshape(-1))/(np.sum((gt_SAT_patient==True).reshape(-1))+np.sum((gt_VAT_patient==True).reshape(-1)))
    false_pos_patient = np.sum((false_pos==True).reshape(-1))/(np.sum((gt_SAT_patient==True).reshape(-1))+np.sum((gt_VAT_patient==True).reshape(-1)))

    return (SAT_GT_SFF, SAT_RES_SFF, SAT_GT_VOL, SAT_RES_VOL,
            med_SAT_DICE_pat,iqr_SAT_DICE_pat,
            mean_SAT_DICE_pat,std_SAT_DICE_pat,
            SAT_VOL_SIM, threeD_SAT_HD,
            threeD_SAT_KAP_pat,
            SAT_sens_pat, SAT_spec_pat,
            VAT_GT_SFF, VAT_RES_SFF, VAT_GT_VOL, VAT_RES_VOL,
            med_VAT_DICE_pat,iqr_VAT_DICE_pat,
            mean_VAT_DICE_pat,std_VAT_DICE_pat,
            VAT_VOL_SIM, threeD_VAT_HD,
            threeD_SAT_DICE_pat, threeD_VAT_DICE_pat,
            threeD_VAT_KAP_pat,
            VAT_sens_pat, VAT_spec_pat,
            false_neg_patient,false_pos_patient)



if test_accuracy == False:
    data = liverfat_validation_loader
else: 
    data = liverfat_test_loader
    
Subject_ID_list = []
for idx in range(max_batch_num_test):
    print(data.dataset[idx]['Subject_ID'])
    Subject_ID_list.append(data.dataset[idx]['Subject_ID'])
    
    
# Segmentation Quality Assessment per subject
# TODO: next time you can actually create a subject class and add these attributes such as SAT_VOL, VAT_VOL etc. 

num_patients = len(Subject_ID_list)
display_on = True
SAT_GT_SFF = np.zeros((num_patients,1))
SAT_RES_SFF = np.zeros((num_patients,1))
SAT_RES_VOL = np.zeros((num_patients,1))
SAT_GT_VOL = np.zeros((num_patients,1))
VAT_GT_SFF = np.zeros((num_patients,1)) 
VAT_RES_SFF = np.zeros((num_patients,1))
VAT_RES_VOL = np.zeros((num_patients,1))
VAT_GT_VOL = np.zeros((num_patients,1))
num_sl = 64

patient_idx = 0
med_SAT_DICE_pat = np.zeros((num_patients,1))
iqr_SAT_DICE_pat = np.zeros((num_patients,1))
mean_SAT_DICE_pat = np.zeros((num_patients,1))
std_SAT_DICE_pat = np.zeros((num_patients,1))
med_VAT_DICE_pat = np.zeros((num_patients,1))
iqr_VAT_DICE_pat = np.zeros((num_patients,1))
mean_VAT_DICE_pat = np.zeros((num_patients,1))
std_VAT_DICE_pat = np.zeros((num_patients,1))
threeD_SAT_DICE_pat = np.zeros((num_patients,1))
threeD_VAT_DICE_pat = np.zeros((num_patients,1))
threeD_SAT_HD = np.zeros((num_patients,1))
threeD_VAT_HD = np.zeros((num_patients,1))
threeD_SAT_KAP_pat = np.zeros((num_patients,1))
threeD_VAT_KAP_pat = np.zeros((num_patients,1))
SAT_VOL_SIM = np.zeros((num_patients,1))
VAT_VOL_SIM = np.zeros((num_patients,1))
SAT_sens_pat = np.zeros((num_patients,1))
SAT_spec_pat = np.zeros((num_patients,1))
VAT_sens_pat = np.zeros((num_patients,1))
VAT_spec_pat = np.zeros((num_patients,1))
false_neg_patient = np.zeros((num_patients,1))
false_pos_patient = np.zeros((num_patients,1))

#for single_patient in patients:
for single_patient in Subject_ID_list:
    print('\nSubject ' + str(single_patient) + ' is processing.')    
    gt_SAT_patient, res_SAT_patient, gt_VAT_patient, res_VAT_patient, fat_image_patient, water_image_patient, eco0_patient = find_patient3D(single_patient, liverfat_transformed_validation_dataset, results_SAT, results_VAT, display_on)
        
    #     gt_SAT_patient, res_SAT_patient, gt_VAT_patient, res_VAT_patient, fat_image_patient, water_image_patient, eco0_patient = find_patient3D(single_patient,liverfat_transformed_test_dataset,results_SAT,gt_SAT,results_VAT,gt_VAT,fat_image, water_image, eco0, display_on)
    
    (SAT_GT_SFF[patient_idx], SAT_RES_SFF[patient_idx], 
     SAT_GT_VOL[patient_idx], SAT_RES_VOL[patient_idx],
     med_SAT_DICE_pat[patient_idx],iqr_SAT_DICE_pat[patient_idx], 
     mean_SAT_DICE_pat[patient_idx],std_SAT_DICE_pat[patient_idx],
     SAT_VOL_SIM[patient_idx], threeD_SAT_HD[patient_idx],
     threeD_SAT_KAP_pat[patient_idx],
     SAT_sens_pat[patient_idx], SAT_spec_pat[patient_idx],
     VAT_GT_SFF[patient_idx], VAT_RES_SFF[patient_idx], 
     VAT_GT_VOL[patient_idx], VAT_RES_VOL[patient_idx], 
     med_VAT_DICE_pat[patient_idx],iqr_VAT_DICE_pat[patient_idx], 
     mean_VAT_DICE_pat[patient_idx],std_VAT_DICE_pat[patient_idx],
     VAT_VOL_SIM[patient_idx], threeD_VAT_HD[patient_idx],
     threeD_SAT_DICE_pat[patient_idx], threeD_VAT_DICE_pat[patient_idx],
     threeD_VAT_KAP_pat[patient_idx],
     VAT_sens_pat[patient_idx], VAT_spec_pat[patient_idx],
     false_neg_patient[patient_idx], false_pos_patient[patient_idx]) = patient_measurements(single_patient,
                                                                                    gt_SAT_patient,
                                                                                    res_SAT_patient,
                                                                                    gt_VAT_patient,
                                                                                    res_VAT_patient,
                                                                                    fat_image_patient,
                                                                                    water_image_patient,
                                                                                    eco0_patient,
                                                                                    threshold,
                                                                                     num_sl)
    
    print(f'\n\t3D SAT-DICE score for subject {single_patient} is {threeD_SAT_DICE_pat[patient_idx]}')
    print(f'\t3D VAT-DICE score for subject {single_patient} is {threeD_VAT_DICE_pat[patient_idx]}')
    
    print(f'\n\t3D false negative score for subject {single_patient} is {false_neg_patient[patient_idx]}')
    print(f'\t3D false positive score for subject {single_patient} is {false_pos_patient[patient_idx]}')

    print(f'\n\t Sensitivity SAT for subject {single_patient} is {SAT_sens_pat[patient_idx]}')
    print(f'\t Sensitivity VAT for subject {single_patient} is {VAT_sens_pat[patient_idx]}')
    
    print(f'\n\t Specificity SAT for subject {single_patient} is {SAT_spec_pat[patient_idx]}')
    print(f'\t Specificity VAT for subject {single_patient} is {VAT_spec_pat[patient_idx]}')
      
    print(f'\n\tAverage SAT-Cohens Kappa for subject {single_patient} is {threeD_SAT_KAP_pat[patient_idx]}')
    print(f'\tAverage VAT-Cohens Kappa for subject {single_patient} is {threeD_VAT_KAP_pat[patient_idx]}')
    
    print(f'\n\tAverage SAT-Hausdorff for subject {single_patient} is {threeD_SAT_HD[patient_idx]}')
    print(f'\tAverage VAT-Hausdorff for subject {single_patient} is {threeD_VAT_HD[patient_idx]}')
    
    print(f'\n\tSAT_VOL_SIM for subject {single_patient} is {SAT_VOL_SIM[patient_idx]}')
    print(f'\tVAT_VOL_SIM for subject {single_patient} is {VAT_VOL_SIM[patient_idx]}')
    
    print(f'\n\tMedian and IQR SAT-DICE score for subject {single_patient} is {med_SAT_DICE_pat[patient_idx]} and {iqr_SAT_DICE_pat[patient_idx]}')
    print(f'\tMedian and IQR VAT-DICE score for subject {single_patient} is {med_VAT_DICE_pat[patient_idx]} and {iqr_VAT_DICE_pat[patient_idx]}')
    
    print(f'\n\tMean and SD SAT-DICE score for subject {single_patient} is {mean_SAT_DICE_pat[patient_idx]} and {std_SAT_DICE_pat[patient_idx]}')
    print(f'\tMean and SD VAT-DICE score for subject {single_patient} is {mean_VAT_DICE_pat[patient_idx]} and {std_VAT_DICE_pat[patient_idx]}')
    
    
    patient_idx = patient_idx + 1 


    
    
print(f'Median and IQR of 3D SAT DICE score: {np.median(threeD_SAT_DICE_pat)} and {scipy.stats.iqr(threeD_SAT_DICE_pat)}')
print(f'Median and IQR of 3D VAT DICE score: {np.median(threeD_VAT_DICE_pat)} and {scipy.stats.iqr(threeD_VAT_DICE_pat)}')
print(f'Mean and STD of 3D SAT DICE score: {np.mean(threeD_SAT_DICE_pat)} and {np.std(threeD_SAT_DICE_pat)}')
print(f'Mean and STD of 3D VAT DICE score: {np.mean(threeD_VAT_DICE_pat)} and {np.std(threeD_VAT_DICE_pat)}\n')

print(f'Mean and STD of false negatives: {np.mean(false_neg_patient)} and {np.std(false_neg_patient)}')
print(f'Mean and STD of false positives: {np.mean(false_pos_patient)} and {np.std(false_pos_patient)}\n')
 
print(f'Mean and STD of Sens. SAT: {np.mean(SAT_sens_pat)} and {np.std(SAT_sens_pat)}')
print(f'Mean and STD of Sens. VAT: {np.mean(VAT_sens_pat)} and {np.std(VAT_sens_pat)}')
print(f'Mean and STD of Spec. SAT: {np.mean(SAT_spec_pat)} and {np.std(SAT_spec_pat)}')
print(f'Mean and STD of Spec. VAT: {np.mean(VAT_spec_pat)} and {np.std(VAT_spec_pat)}\n')

print(f'Median and IQR of of Sens. SAT: {np.median(SAT_sens_pat)} and {scipy.stats.iqr(SAT_sens_pat)}')
print(f'Median and IQR of of Sens. VAT: {np.median(VAT_sens_pat)} and {scipy.stats.iqr(VAT_sens_pat)}')
print(f'Median and IQR of Spec. SAT: {np.median(SAT_spec_pat)} and {scipy.stats.iqr(SAT_spec_pat)}')
print(f'Median and IQR of Spec. VAT: {np.median(VAT_spec_pat)} and {scipy.stats.iqr(VAT_spec_pat)}\n')

print(f'Median and IQR of SAT Cohens Kappa: {np.median(threeD_SAT_KAP_pat)} and {scipy.stats.iqr(threeD_SAT_KAP_pat)}')
print(f'Median and IQR of VAT Cohens Kappa: {np.median(threeD_VAT_KAP_pat)} and {scipy.stats.iqr(threeD_VAT_KAP_pat)}')
print(f'Mean and STD of SAT Cohens Kappa: {np.mean(threeD_SAT_KAP_pat)} and {np.std(threeD_SAT_KAP_pat)}')
print(f'Mean and STD of VAT Cohens Kappa: {np.mean(threeD_VAT_KAP_pat)} and {np.std(threeD_VAT_KAP_pat)}\n')

print(f'Median and IQR of SAT Volume Similarity: {np.median(SAT_VOL_SIM)} and {scipy.stats.iqr(SAT_VOL_SIM)}')
print(f'Median and IQR of VAT Volume Similarity: {np.median(VAT_VOL_SIM)} and {scipy.stats.iqr(VAT_VOL_SIM)}')
print(f'Mean and STD of SAT Volume Similarity: {np.mean(SAT_VOL_SIM)} and {np.std(SAT_VOL_SIM)}')
print(f'Mean and STD of VAT Volume Similarity: {np.mean(VAT_VOL_SIM)} and {np.std(VAT_VOL_SIM)}\n')

print(f'Median and IQR of SAT Volume Distance: {np.median(1-SAT_VOL_SIM)} and {scipy.stats.iqr(1-SAT_VOL_SIM)}')
print(f'Median and IQR of VAT Volume Distance: {np.median(1-VAT_VOL_SIM)} and {scipy.stats.iqr(1-VAT_VOL_SIM)}')
print(f'Mean and STD of SAT Volume Distance: {np.mean(1-SAT_VOL_SIM)} and {np.std(1-SAT_VOL_SIM)}')
print(f'Mean and STD of VAT Volume Distance: {np.mean(1-VAT_VOL_SIM)} and {np.std(1-VAT_VOL_SIM)}\n')

print('Processing done!')


SAT_VOL_ICC = icc(np.concatenate((SAT_GT_VOL,SAT_RES_VOL), axis=1), model='twoway', type='agreement', unit='single')
VAT_VOL_ICC = icc(np.concatenate((VAT_GT_VOL,VAT_RES_VOL), axis=1), model='twoway', type='agreement', unit='single')

print(SAT_VOL_ICC)
print(VAT_VOL_ICC)
    
with open(main_dir_out_files+main_filename+str(train_batch_size)+str(num_epochs)+str(lr)+str(loss_threshold)+'-val-paper.obj', 'wb') as fp:
    pickle.dump({'threeD_SAT_DICE_pat':[ele[0] for ele in threeD_SAT_DICE_pat], 
                 'threeD_VAT_DICE_pat':[ele[0] for ele in threeD_VAT_DICE_pat], 
                 'threeD_SAT_KAP_pat':[ele[0] for ele in threeD_SAT_KAP_pat], 
                 'threeD_VAT_KAP_pat':[ele[0] for ele in threeD_VAT_KAP_pat],
                 'SAT_VOL_SIM': [ele[0] for ele in SAT_VOL_SIM],
                 'VAT_VOL_SIM': [ele[0] for ele in VAT_VOL_SIM],
                 'SAT_VOL_ICC': SAT_VOL_ICC,
                 'VAT_VOL_ICC': VAT_VOL_ICC,
                 'FN': [ele[0] for ele in false_neg_patient],
                 'FP': [ele[0] for ele in false_pos_patient]}, fp)

    
    