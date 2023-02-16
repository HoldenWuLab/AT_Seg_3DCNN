import numpy as np
import torch
from losses import LogWeightedDICELossMultiClass3D
from models_all import ACD3DUNet
import torch.optim as optim
import os
from tqdm import tqdm
from ATSeg_load_HAT_3D import load_LiverFatDataset
import time
import GPUtil

# select the free GPU automatically
gpu_id = GPUtil.getFirstAvailable(order = 'first', maxLoad=0.05, maxMemory=0.40, attempts=600, interval=60, verbose=False)[0]
print(f'The GPU ID is: {gpu_id}')
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) 

# General variables to define
train_batch_size = 1
test_batch_size = 1
use_gpu = True
lr = 0.0005
num_epochs = 25
loss_threshold = 0.5
loss_list = np.zeros((num_epochs))
val_loss_list = np.zeros((num_epochs))
save_vat = 'VATMasks'
save_sat = 'SATMasks'
save_bg = 'BGMasks'
save = 'ALLMasks'
test_accuracy = True
train_accuracy = False

# Set up model, optimizer and loss
model = ACD3DUNet(3,3)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = LogWeightedDICELossMultiClass3D() 
t = torch.cuda.get_device_properties(0).total_memory

# please update the directories accordingly
main_dir_out_files = '/skafali/AT Seg/out-files/'
main_filename = 'ACD3DUNET-FBDL'

# Training function
def train(epoch):

    model.train()
    max_batch_num = len(liverfat_transformed_train_dataset)
    print(f"max number of batches is: {max_batch_num}")
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
            batch_data['image'][batch_idx, :, :, :, :] = temp_batch_data['image']
            batch_data['mask'][batch_idx, :, :, :, :] = temp_batch_data['mask']
            
        # constructing the images and masks according to the batch size
        image = batch_data['image']
        mask = batch_data['mask']
             
        if use_gpu:
            image, mask = image.cuda(), mask.cuda()

        image = image.float()
        mask = mask.float()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        
        output = model(image)
        loss = criterion(output, mask, loss_threshold)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss: {:.6f}'.format(
            epoch, batch_number * len(image), str(max_batch_num),
            100. * batch_number / max_batch_num, loss.item()))

    return output, mask, loss


# Testing function
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
        
        image = image[np.newaxis, :, :, :]
        mask = mask[np.newaxis, :, :, :]
        
        if use_gpu:
            image, mask = image.cuda(), mask.cuda()

        image = image.float()
        mask = mask.float()
        output = model(image)
        # maxes, out = torch.max(output, 1, keepdim=True)

        # output SAT, VAT and BG masks
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-outs.npy'.format(save_vat,
                                                                                batch_idx),
                output[:, 0, :, :, :].data.cpu().numpy())
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-outs.npy'.format(save_sat,
                                                                                batch_idx),
                output[:, 1, :, :, :].data.cpu().numpy())
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-outs.npy'.format(save_bg,
                                                                                batch_idx),
                output[:, 2, :, :, :].data.cpu().numpy())
        # ground truth masks
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-masks.npy'.format(save_vat,
                                                                                 batch_idx),
                mask[:, 0, :, :, :].data.byte().cpu().numpy())
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-masks.npy'.format(save_sat,
                                                                                 batch_idx),
                mask[:, 1, :, :, :].data.byte().cpu().numpy())
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-masks.npy'.format(save_bg,
                                                                                 batch_idx),
                mask[:, 2, :, :, :].data.byte().cpu().numpy())

        # input images
        np.save(main_dir_out_files+main_filename+'-{}-batch-{}-images.npy'.format(save,
                                                                                  batch_idx),
                image.data.float().cpu().numpy())

        test_loss += criterion(output, mask, loss_threshold).item()

    # Average Loss
    test_loss /= len(loader)
    print('\nTest Set: Average Loss: {:.4f})\n'.format(
        test_loss))
        
        
# Calculate the number of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_param = count_parameters(model)
print("Number of parameters for ACD 3D UNet is: " + str(num_param))


# Start Training
ii = 0
for ep in tqdm(range(num_epochs)):
    t0 = time.time()
    liverfat_transformed_train_dataset, liverfat_transformed_validation_dataset, liverfat_transformed_test_dataset, liverfat_train_loader, liverfat_validation_loader, liverfat_test_loader = load_LiverFatDataset(
        train_batch_size, test_batch_size)
    output, mask, loss = train(ii)
    t_end = time.time() - t0
    t_end = t_end / 60
    print('{} minutes for an epoch'.format(t_end))
    loss_list[ii] = loss.cpu().detach().numpy()
    torch.cuda.empty_cache()
    with torch.no_grad():
        # validation or test loss
        test(liverfat_train_loader, liverfat_validation_loader, liverfat_test_loader,train_accuracy=train_accuracy, test_accuracy=test_accuracy)
    ii = ii + 1
    # save every 5 epochs
    if ii % 5 == 0:
        torch.save(model.state_dict(),
                   main_dir_out_files + main_filename + '-{}-{}-{}-{}-{}-{}'.format(train_batch_size,
                                                                                    num_epochs,
                                                                                    lr,
                                                                                    'ALL',
                                                                                    loss_threshold,
                                                                                    ii))
