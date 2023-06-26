import argparse

import datetime
import json
import numpy
import os
import torch
import torchvision

from tqdm import tqdm


import core

from core.augmentations import Compose, AddNoise, RandomRotate, RandomVerticallyFlip
from core.loader import data_folders
from core.metrics import RunningScore
from core.utils import importName, np_to_tb, append_filter, EarlyStopper


# Fix the random seeds: 
numpy.random.seed(seed=2022)
torch.backends.cudnn.deterministic = True
torch.manual_seed(2022)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(2022)


class CustomSampler(torch.utils.data.Sampler):
    def __init__(self, sample_list):
        self.sample_list = sample_list
        
    def __iter__(self):
        char = ['i' if numpy.random.randint(2) == 1 else 'x']
        self.indices = [idx for (idx, name) in enumerate(self.sample_list) if char[0] in name]
        return (self.indices[i] for i in torch.randperm(len(self.indices)))


def train(args):
    print(args)
    # Importing class and methods at run time
    section_dataset = importName(f'core.loader.dataset_{args.dataset}', 'section_dataset')
    split_train_val = importName(f'core.loader.dataset_{args.dataset}', 'split_train_val')

    # Obtaining data folder and setting CPU/GPU device
    data_folder = data_folders[args.dataset]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Generate the train and validation sets for the model:
    split_train_val(args, data_folder, per_val=args.per_val)

    # Setup log files 
    border_factor = args.border_factor if args.loss_function in ['abl','dtl'] else '0.0'
    running_name = f'{args.dataset}_{args.architecture}_D={args.channel_delta}_F={args.filter}_L={args.loss_function}_B={border_factor}_M={args.batch_size}_P={args.per_val}{"_A" if args.aug else ""}{"_W" if args.class_weights else ""}'
    subfolders = [subfolder.path for subfolder in os.scandir('runs_section') if subfolder.is_dir()]
    if any(running_name in substr for substr in subfolders):
        print(f'[WARNING] Folder containing {running_name} already exists. Quitting!')
        print('---'*20, '\n')
        return
    else:
        folder_path = os.path.join(args.save_folder, f'{running_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.mkdir(folder_path)
    
    # Setup augmentations
    if args.aug:
        print('Data Augmentation ON.')
        data_aug = Compose([RandomRotate(degree=10), RandomVerticallyFlip(), AddNoise()])
    else:
        print('Data Augmentation OFF.')
        data_aug = None

    # Traning accepts augmentation, unlike validation:
    train_set = section_dataset(channel_delta=args.channel_delta, data_folder=data_folder, split='train', is_transform=True, augmentations=data_aug)
    valid_set = section_dataset(channel_delta=args.channel_delta, data_folder=data_folder, split='val',   is_transform=True)

    # Get classes
    n_classes = train_set.n_classes
    class_names = train_set.get_class_names()

    # Create sampler:
    with open(os.path.join(data_folder, 'splits', 'section_train.txt'), 'r') as file_buffer:
        train_list = file_buffer.read().splitlines()
    with open(os.path.join(data_folder, 'splits', 'section_val.txt'), 'r') as file_buffer:
        val_list = file_buffer.read().splitlines()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=CustomSampler(train_list), num_workers=0, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, sampler=CustomSampler(val_list),   num_workers=0)

    # Setup Model
    if args.resume is None:
        if args.channel_delta > 0 and args.filter != 'None':
            raise ValueError('Multiple channels and attached filter cannot run jointly.')
        n_channels = 3 if args.channel_delta > 0 else 2 if (args.channel_delta == 0 and args.filter != 'None') else 1
        model = getattr(core.models, core.models.architectures[args.architecture])(n_channels=n_channels, n_classes=n_classes)
        # model = core.models.get_model(args.architecture, True, n_channels, n_classes)
        print(f'Creating Model {args.architecture.upper()} with {n_channels} input channels, delta={args.channel_delta} and filter={args.filter}')
    else:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            model = torch.load(args.resume)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    model = model.to(device)

    # Weights are inversely proportional to the frequency of the classes in the training set
    if args.class_weights:
        print('Weighted Loss Enabled.') 
        class_weights = torch.tensor(train_set.get_class_weights(), device=device, requires_grad=False)
    else:
        print('Weighted Loss Disabled.') 
        class_weights = None

    # Instantiating criterion and optimizer
    loss_map = {
        'cel':('CrossEntropyLoss',      {'reduction':'sum', 'weight':class_weights}),
        'abl':('ActiveBoundaryLoss',    {'border_factor':args.border_factor, 'device':device, 'weight':class_weights}),
        'abv':('ActiveBoundaryLoss_',   {'device':device, 'reduction':'sum', 'weight':class_weights}),
        'dtl':('DistanceTransformLoss', {'border_factor':args.border_factor, 'reduction':'sum', 'weight':class_weights})
    }
    loss_name, loss_args = loss_map[args.loss_function]
    criterion = getattr(core.loss, loss_name)(**loss_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, amsgrad=True)
    # optimizer = torch.optim.Adadelta(model.parameters())

    # Setup metrics and early stopping
    early_stopper     = EarlyStopper(tolerance=int(0.1*args.n_epoch))
    running_metrics_T = RunningScore(n_classes, threshold=2)
    running_metrics_V = RunningScore(n_classes, threshold=2)

    # training
    img_processor = None
    for epoch in range(args.n_epoch):
        # Training Mode:
        model.train()
        loss_train, batch_counter = 0, 0
        
        print(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        for batch, (images, labels) in enumerate(train_loader):
            image_original, labels_original = images, labels
            images, labels = images.to(device), labels.to(device)
            if args.filter != 'None':
                if img_processor is None: img_processor = append_filter(args.filter, device=device)
                images = img_processor(images, concat=True)

            optimizer.zero_grad()
            outputs = model(images)
            
            print("\nOUTPUTS:")
            print(type(outputs))
            
            running_metrics_T.update(slices=outputs, targets=labels)

            loss = criterion(slices=outputs, targets=labels)
            loss_train += loss.item()
            loss.backward()

            # gradient clipping
            if args.clip != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            batch_counter = batch_counter + 1

            numbers = [0]
            if batch in numbers:
                # number 0 image in the batch
                tb_original_image = torchvision.utils.make_grid(image_original[0][0], normalize=True, scale_each=True)

                labels_original = labels_original.numpy()[0]
                correct_label_decoded = train_set.decode_segmap(numpy.squeeze(labels_original))
                out = torch.nn.functional.softmax(outputs, dim=1)

                # this returns the max. channel number:
                prediction = out.max(1)[1].cpu().numpy()[0]
                # this returns the confidence:
                confidence = out.max(1)[0].cpu().detach()[0]
                tb_confidence = torchvision.utils.make_grid(confidence, normalize=True, scale_each=True)

                decoded = train_set.decode_segmap(numpy.squeeze(prediction))

                unary = outputs.cpu().detach()
                unary_max = torch.max(unary)
                unary_min = torch.min(unary)
                unary = unary.add((-1*unary_min))
                unary = unary/(unary_max - unary_min)

                for channel in range(0, len(class_names)):
                    decoded_channel = unary[0][channel]
                    tb_channel = torchvision.utils.make_grid(decoded_channel, normalize=True, scale_each=True)

        # Average metrics
        loss_train /= batch_counter
        score = running_metrics_T.get_scores(epoch+1)
        running_metrics_T.reset()
        print(f'Epoch [{epoch+1}/{args.n_epoch}] training Loss: {loss_train:.4f}')

        # Validation Mode:
        if args.per_val > 0.:
            with torch.no_grad():  # operations inside don't track history
                model.eval()
                loss_valid, batch_counter = 0, 0

                for batch, (images_val, labels_val) in tqdm(enumerate(valid_loader)):
                    image_original, labels_original = images_val, labels_val
                    images_val, labels_val = images_val.to(device), labels_val.to(device)
                    if args.filter != 'None':
                        if img_processor is None: img_processor = append_filter(args.filter, device=device)
                        images_val = img_processor(images_val, concat=True)

                    outputs_val = model(images_val)
                    running_metrics_V.update(slices=outputs_val, targets=labels_val)
                    
                    loss = criterion(slices=outputs_val, targets=labels_val)
                    loss_valid += loss.item()
                    batch_counter = batch_counter + 1

                    numbers = [0]
                    if batch in numbers:
                        # number 0 image in the batch
                        tb_original_image = torchvision.utils.make_grid(
                            image_original[0][0], normalize=True, scale_each=True)
                        labels_original = labels_original.numpy()[0]
                        correct_label_decoded = train_set.decode_segmap(numpy.squeeze(labels_original))

                        out = torch.nn.functional.softmax(outputs_val, dim=1)

                        # this returns the max. channel number:
                        prediction = out.max(1)[1].cpu().detach().numpy()[0]
                        # this returns the confidence:
                        confidence = out.max(1)[0].cpu().detach()[0]
                        tb_confidence = torchvision.utils.make_grid(confidence, normalize=True, scale_each=True)

                        decoded = train_set.decode_segmap(numpy.squeeze(prediction))

                        unary = outputs.cpu().detach()
                        unary_max, unary_min = torch.max(unary), torch.min(unary)
                        unary = unary.add((-1*unary_min))
                        unary = unary/(unary_max - unary_min)

                        for channel in range(0, len(class_names)):
                            tb_channel = torchvision.utils.make_grid(unary[0][channel], normalize=True, scale_each=True)

                # Average metrics
                loss_valid /= batch_counter
                print(f'Epoch [{epoch+1}/{args.n_epoch}] validation Loss: {loss_valid:.4f}')
                score = running_metrics_V.get_scores(epoch+1)
                for key, value in score.items(): 
                    print(f'\t{key}: {value}')
                running_metrics_V.reset()

                if early_stopper.stop(current_score=score['Decision']) and (epoch+1)/args.n_epoch >= 0.5:
                    print(f'[WARNING] Early-stopping at epoch {epoch+1} activated with Decision={early_stopper.get_score()}. Quitting!')
                    print('---'*20, '\n')
                    break
                elif score['Decision'] >= early_stopper.get_score():
                    score['Valid loss'] = loss_valid
                    torch.save(model, os.path.join(folder_path, 'model.pth'))
                    torch.save(score, os.path.join(folder_path, 'score.pth'))
                    torch.save(vars(args), os.path.join(folder_path, "parameters.pth"))
                    with open(os.path.join(folder_path, 'score_train.json'), 'w') as json_buffer:
                        json.dump(score, json_buffer, indent=4)
                    print(f'[BEST] Model Saved: {early_stopper.get_score()}')
                print()
        else:  
            # When validation is off: save latest model
            if (epoch+1) % 10 == 0:
                torch.save(model, os.path.join(folder_path, "model_ep{epoch+1}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    
    parser.add_argument('--architecture',  type=str,   default='deconvnet',    help='Architecture to use [\'deconvnet, segnet, unet, danet3, hrnet\']', choices=['all', 'deconvnet', 'segnet', 'unet', 'danet3', 'hrnet'])
    parser.add_argument('--batch_size',    type=int,   default=12,             help='Batch Size')
    parser.add_argument('--channel_delta', type=int,   default=0,              help='Number of variable input channels')
    parser.add_argument('--device',        type=str,   default='cuda:0',       help='Cuda device or cpu execution')
    parser.add_argument('--filter',        type=str,   default='None',         help='Add filter as an extra channel/layer', choices=['None', 'canny', 'gabor', 'sobel'])
    parser.add_argument('--loss_function', type=str,   default='abl',          help='Loss function to use [\'ABL, ABV, CEL\']', choices=['abl','abv','cel','dtl'])
    parser.add_argument('--border_factor', type=float, default=0.1,            help='Weight to multiply the border loss [\'0.1-1.0\']')
    parser.add_argument('--n_epoch',       type=int,   default=60,             help='# of the epochs')

    parser.add_argument('--save_folder',   type=str,   default='runs_section', help='Folder storing execution outputs')
    parser.add_argument('--dataset',       type=str,   default='NL',           help='Name of the adopted dataset: NL (Netherlands F3 Block), NS (Nova Scotia Penobscot), NZ (New Zealand Petroleum).', choices=['NL', 'NS', 'NZ'])
    parser.add_argument('--resume',        type=str,   default=None,           help='Path to previous saved model to restart from')
    parser.add_argument('--clip',          type=float, default=0.0,            help='Max norm of the gradients if clipping. Set to zero to disable. ')
    parser.add_argument('--per_val',       type=float, default=0.10,           help='Percentage of the training data for validation')
    parser.add_argument('--pretrained',    type=bool,  default=False,          help='Pretrained models not supported. Keep as False for now.')
    parser.add_argument('--aug',           action='store_true',                help='Whether to use data augmentation.')
    parser.add_argument('--class_weights', action='store_true',                help='Whether to use class weights to reduce the effect of class imbalance')

    local_params = [
        '--architecture', 'deconvnet',
        '--batch_size', '1',
        '--channel_delta', '0',
        '--device', 'cpu',
        '--filter', 'None',
        '--loss_function', 'dtl',
        '--border_factor', '0.1',
        '--n_epoch', '10',
        
        '--dataset', 'NL',
        '--clip', '0.1',
        '--per_val', '0.1',
        '--aug', 
        '--class_weights'
    ]

    args = parser.parse_args(args=None)
    if args.architecture != 'all':
        train(args)
    else:
        for arch in ['deconvnet','segnet','unet', 'danet-3', 'hrnet']:
            args.architecture = arch
            train(args)
