import argparse

import numpy
import os
import torch
from core.loader.dataset_NL import patch_dataset
from core.models.danet.danet3 import DaNet3
import torchvision

from datetime import datetime
from tqdm import tqdm

import itertools
from sklearn.model_selection import train_test_split

import core.loss
import core.models

from core.augmentations import Compose, AddNoise, RandomHorizontallyFlip, RandomRotate, RandomVerticallyFlip
from core.metrics import RunningScore
from core.models import get_model
from core.utils import np_to_tb

# Fix the random seeds: 
torch.backends.cudnn.deterministic = True
torch.manual_seed(2019)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed_all(2019)
numpy.random.seed(seed=2019)

def split_train_val(args, per_val=0.1):
    # create inline and crossline pacthes for training and validation:
    loader_type = 'patch'
    labels = numpy.load(os.path.join('data', 'train', 'train_labels.npy'))
    iline, xline, depth = labels.shape

    # INLINE PATCHES: ------------------------------------------------
    i_list = []
    horz_locations = range(0, xline-args.stride, args.stride)
    vert_locations = range(0, depth-args.stride, args.stride)
    for i in range(iline):
        # for every inline:
        # images are references by top-left corner:
        locations = [[j, k] for j in horz_locations for k in vert_locations]
        patches_list = ['i_'+str(i)+'_'+str(j)+'_'+str(k) for j, k in locations]
        i_list.append(patches_list)

    # flatten the list
    i_list = list(itertools.chain(*i_list))

    # XLINE PATCHES: ------------------------------------------------
    x_list = []
    horz_locations = range(0, iline-args.stride, args.stride)
    vert_locations = range(0, depth-args.stride, args.stride)
    for j in range(xline):
        # for every xline:
        # images are references by top-left corner:
        locations = [[i, k] for i in horz_locations for k in vert_locations]
        patches_list = ['x_'+str(i)+'_'+str(j)+'_'+str(k) for i, k in locations]
        x_list.append(patches_list)

    # flatten the list
    x_list = list(itertools.chain(*x_list))

    list_train_val = i_list + x_list

    # create train and test splits:
    list_train, list_val = train_test_split(
        list_train_val, test_size=per_val, shuffle=True)

    # write to files to disK:
    file_object = open(os.path.join('data', 'splits', loader_type + '_train_val.txt'), 'w')
    file_object.write('\n'.join(list_train_val))
    file_object.close()
    file_object = open(os.path.join('data', 'splits', loader_type + '_train.txt'), 'w')
    file_object.write('\n'.join(list_train))
    file_object.close()
    file_object = open(os.path.join('data', 'splits', loader_type + '_val.txt'), 'w')
    file_object.write('\n'.join(list_val))
    file_object.close()


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Generate the train and validation sets for the model:
    split_train_val(args, per_val=args.per_val)

    current_time = datetime.now().strftime('%b%d_%H%M%S')
    log_dir = os.path.join('runs_patch', current_time + "_{}".format(args.arch))
    # Setup Augmentations
    if args.aug:
        data_aug = Compose([RandomRotate(10), RandomHorizontallyFlip(), AddNoise()])
    else:
        data_aug = None

    train_set = patch_dataset(is_transform=True, split='train', stride=args.stride, patch_size=args.patch_size, augmentations=data_aug)
    valid_set = patch_dataset(is_transform=True, split='val', stride=args.stride, patch_size=args.patch_size)

    n_classes = train_set.n_classes

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True)
    valloader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, num_workers=4)

    # Setup Metrics
    running_metrics = RunningScore(n_classes)
    running_metrics_val = RunningScore(n_classes)

    # Setup Model
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            model = torch.load(args.resume)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    else:
        # model = get_model(args.arch, args.pretrained, n_classes)
        model = DaNet3(classes=n_classes)
        print("Instanciou a danet")

    # Use as many GPUs as we can
    model = torch.nn.DataParallel(
        model, device_ids=range(torch.cuda.device_count()))
    model = model.to(device)  # Send to GPU

    # PYTROCH NOTE: ALWAYS CONSTRUCT OPTIMIZERS AFTER MODEL IS PUSHED TO GPU/CPU,

    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        print('Using custom optimizer')
        optimizer = model.module.optimizer
    else:
        # optimizer = torch.optim.Adadelta(model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)

    # Weights are inversely proportional to the frequency of the classes in the training set
    if args.class_weights:
        class_weights = torch.tensor(train_set.get_class_weights(), device=device, requires_grad=False)
    else:
        class_weights = None
    criterion = core.loss.CrossEntropyLoss(weight=class_weights)

    best_iou = -100.0
    class_names = train_set.get_class_names()


    # training
    print("Começando o treinamento")
    for epoch in range(args.n_epoch):
        # Training Mode:
        model.train()
        loss_train, total_iteration = 0, 0

        for i, (images, labels) in enumerate(trainloader):
            image_original, labels_original = images, labels
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            print("Gerou os outputs:")
            print(type(outputs))

            pred = outputs.detach().max(1)[1].cpu().numpy()
            gt = labels.detach().cpu().numpy()
            running_metrics.update(gt, pred)

            loss = criterion(input=outputs, target=labels, weight=class_weights)
            loss_train += loss.item()
            loss.backward()

            # gradient clipping
            if args.clip != 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            total_iteration = total_iteration + 1

            if (i) % 20 == 0:
                print("Epoch [%d/%d] training Loss: %.4f" % (epoch + 1, args.n_epoch, loss.item()))

            numbers = [0]
            if i in numbers:
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
        loss_train /= total_iteration
        score, class_iou = running_metrics.get_scores()
        running_metrics.reset()

        # Validation Mode:
        if args.per_val != 0:
            with torch.no_grad():  # operations inside don't track history
                model.eval()
                loss_val, total_iteration_val = 0, 0

                for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                    image_original, labels_original = images_val, labels_val
                    images_val, labels_val = images_val.to(device), labels_val.to(device)

                    outputs_val = model(images_val)
                    pred = outputs_val.detach().max(1)[1].cpu().numpy()
                    gt = labels_val.detach().cpu().numpy()

                    running_metrics_val.update(gt, pred)

                    loss = criterion(input=outputs_val, target=labels_val)

                    total_iteration_val = total_iteration_val + 1

                    if (i_val) % 20 == 0:
                        print("Epoch [%d/%d] validation Loss: %.4f" % (epoch, args.n_epoch, loss.item()))

                    numbers = [0]
                    if i_val in numbers:
                        # number 0 image in the batch
                        tb_original_image = torchvision.utils.make_grid(image_original[0][0], normalize=True, scale_each=True)
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

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)


                running_metrics_val.reset()

                if score['Mean IoU'] >= best_iou:
                    best_iou = score['Mean IoU']
                    torch.save(model, os.path.join(log_dir, f"{args.arch}_model.pth"))
        # Validation turned off: save latest model
        else: 
            if (epoch+1) % 5 == 0:
                torch.save(model, os.path.join(log_dir, f"{args.arch}_ep{epoch+1}_model.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    # parser.add_argument('--arch', nargs='?', type=str, default='patch_deconvnet', help='Architecture to use [\'patch_deconvnet, path_deconvnet_skip, section_deconvnet, section_deconvnet_skip\']')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=60, help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16, help='Batch Size')
    parser.add_argument('--resume', nargs='?', type=str, default=None, help='Path to previous saved model to restart from')
    parser.add_argument('--clip', nargs='?', type=float, default=0.1, help='Max norm of the gradients if clipping. Set to zero to disable. ')
    parser.add_argument('--per_val', nargs='?', type=float, default=0.2, help='percentage of the training data for validation')
    parser.add_argument('--stride', nargs='?', type=int, default=40, help='The vertical and horizontal stride when we are sampling patches from the volume. The smaller the better, but the slower the training is.')
    parser.add_argument('--patch_size', nargs='?', type=int, default=40, help='The size of each patch')
    parser.add_argument('--pretrained', nargs='?', type=bool, default=False, help='Pretrained models not supported. Keep as False for now.')
    parser.add_argument('--aug', nargs='?', type=bool, default=False, help='Whether to use data augmentation.')
    parser.add_argument('--class_weights', nargs='?', type=bool, default=False, help='Whether to use class weights to reduce the effect of class imbalance')

    args = parser.parse_args()
    train(args)
