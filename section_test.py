import argparse

import datetime
import json
import numpy
import os
import torch
import torchvision

import core.models

from core.loader import data_folders
from core.metrics import RunningScore
from core.utils import importName, np_to_tb, append_filter


def test(args):
    print(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    
    # Importing class and methods at run time
    section_dataset = importName(f'core.loader.dataset_{args.dataset}', 'section_dataset')
    split_test_eval = importName(f'core.loader.dataset_{args.dataset}', 'split_test_eval')
    
    # Logging setup
    model_path = os.path.join(args.folder_path, 'model.pth')
    param_path = os.path.join(args.folder_path, 'parameters.pth')

    if not os.path.exists(model_path):
        print(f'[WARNING] Folder {running_name} does NOT contain model.pth. Quitting!')
        print('---'*20, '\n')
        return

    # Load parameters when available
    param = torch.load(param_path) if os.path.exists(param_path) else None
    if param is not None:
        args.channel_delta = param['channel_delta']
        args.device        = param['device']
        args.filter        = param['filter']
        args.per_val       = param['per_val']
    print(args)

    # Obtain data folder and setting CPU/GPU device
    data_folder = data_folders[args.dataset]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model:
    model = torch.load(model_path, map_location=device)
    model = model.to(device)

    running_metrics = None
    two_stream = None # type(model) is core.models.DeconvNetTS

    if 'both' in args.split: splits = ['test1', 'test2']
    else: splits = [args.split]

    for split in splits:
        split_test_eval(args, data_folder, split, per_val=args.per_val)

        test_set = section_dataset(channel_delta=args.channel_delta, data_folder=data_folder, split=split, is_transform=True, augmentations=None)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False)
        class_names = test_set.get_class_names()

        # instantiate overall and split metric objects:
        # partial_metrics = RunningScore(test_set.n_classes)
        if running_metrics is None:
            running_metrics = RunningScore(test_set.n_classes, threshold=2)

        # testing mode:
        img_processor = None
        with torch.no_grad():  # operations inside don't track history
            model.eval()
            total_iteration = 0
            for batch, (images, labels) in enumerate(test_loader):
                total_iteration = total_iteration + 1
                image_original, labels_original = images, labels
                if two_stream:
                    gabors = detect_gabor_edges(images, frequency=0.1)
                    images, gabors, labels = images.to(device), gabors.to(device), labels.to(device)
                    outputs = model(images, gabors)
                else:
                    images, labels = images.to(device), labels.to(device)
                    if args.filter != 'None':
                        if img_processor is None: img_processor = append_filter(args.filter, device=device)
                        images = img_processor(images, concat=True)
                    outputs = model(images)
                
                running_metrics.update(slices=outputs, targets=labels)
                numbers = test_set.get_prediction_horizons()

                if batch in numbers:
                    tb_original_image = torchvision.utils.make_grid(image_original[0][0], normalize=True, scale_each=True)

                    labels_original = labels_original.numpy()[0]
                    correct_label_decoded = test_set.decode_segmap(numpy.squeeze(labels_original), save_name=os.path.join(args.folder_path, f'plot_grount_truth_{split}_{str(batch)}.png'))
                    out = torch.nn.functional.softmax(outputs, dim=1)

                    # this returns the max. channel number:
                    prediction = out.max(1)[1].cpu().numpy()[0]
                    # this returns the confidence:
                    confidence = out.max(1)[0].cpu().detach()[0]
                    tb_confidence = torchvision.utils.make_grid(confidence, normalize=True, scale_each=True)

                    decoded = test_set.decode_segmap(numpy.squeeze(prediction), save_name=os.path.join(args.folder_path, f'plot_predictions_{split}_{str(batch)}.png'))

                    # uncomment if you want to visualize the different class heatmaps
                    unary = outputs.cpu().detach()
                    unary_max = torch.max(unary)
                    unary_min = torch.min(unary)
                    unary = unary.add((-1*unary_min))
                    unary = unary/(unary_max - unary_min)

                    for channel in range(0, len(class_names)):
                        decoded_channel = unary[0][channel]
                        tb_channel = torchvision.utils.make_grid(decoded_channel, normalize=True, scale_each=True)

        # get scores
        # score, class_iou = partial_metrics.get_scores()
        # partial_metrics.reset()

    # FINAL TEST RESULTS:
    score = running_metrics.get_scores()

    print('--------------- FINAL RESULTS -----------------')
    print(f'BF1 Score: {score["BF1 Score"]:.3f}')
    print(f'Pixel Acc: {score["Pixel Acc"]:.3f}')
    for cdx, class_name in enumerate(class_names):
        print(f'     {class_name}_accuracy {score["Class Accuracy"][cdx]:.3f}')
    print(f'Mean Class Acc: {score["Mean Class Acc"]:.3f}')
    print(f'Freq Weighted IoU: {score["Freq Weighted IoU"]:.3f}')
    print(f'Mean IoU: {score["Mean IoU"]:0.3f}')
    print('-----------------------------------------------\n')
    
    with open(os.path.join(args.folder_path, 'score_test.json'), 'w') as json_buffer:
        json.dump(score, json_buffer, indent=4)
    
    # Save confusion matrix: 
    numpy.savetxt(os.path.join(args.folder_path,'confusion.csv'), score['Confusion Matrix'], delimiter=' ')
    numpy.save(os.path.join(args.folder_path,'score'), score, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')

    parser.add_argument('--folder_path',   type=str,             default='all',  help='Path to the saved model')
    parser.add_argument('--channel_delta', type=int,             default=0,      help='# of variable input channels')
    parser.add_argument('--device',        type=str,             default='cpu',  help='Cuda device or cpu execution')
    parser.add_argument('--filter',        type=str,             default='None', help='Add filter as an extra channel/layer', choices=['None', 'gabor','hessian', 'sobel'])
    
    parser.add_argument('--architecture',  type=str,             default='deconvnet', help='Architecture to use [\'deconvnet, segnet, unet\']', choices=['deconvnet','segnet','unet'])
    parser.add_argument('--dataset',       type=str,             default='NL',   help='Name of the adopted dataset: NL (Netherlands F3 Block), NS (Nova Scotia Penobscot), NZ (New Zealand Petroleum).', choices=['NL', 'NS', 'NZ'])
    parser.add_argument('--crossline',     type=bool, nargs='?', default=True,   help='whether to test in crossline mode')
    parser.add_argument('--inline',        type=bool, nargs='?', default=True,   help='whether to test inline mode')
    parser.add_argument('--split',         type=str,  nargs='?', default='both', help='Choose from: "test1", "test2", "both", or "eval" to change which region to test on',  choices=['both', 'eval', 'test1', 'test2'])
    
    local_params = [
        '--folder_path', 'runs_section/NL_deconvnet_D=0_F=None_L=cel_P=0.1_W_20230421_150926/',
        '--channel_delta', '0',
        '--dataset', 'NL',
        '--device', 'cuda:0',
        '--filter', 'gabor',
        '--split', 'both',
    ]

    args = parser.parse_args(args=None)
    if args.folder_path != 'all':
        test(args)
    else:
        for root, folders, files in os.walk('./runs_section/', topdown=False):
            for folder in sorted(folders):
                if folder.startswith(f'{args.dataset}'):
                    args.folder_path = os.path.join(root, folder)
                    test(args)
    