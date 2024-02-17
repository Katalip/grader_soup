import os
import torch
import numpy as np
import argparse
import datetime
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.dist_dependence_measures import distance_correlation
from utils.metrics import (
    get_dice_threshold,
    get_soft_dice,
    get_uncertainty_metrics
)
from models.bulid import build_model
from dataset.get_dataset import getDataset
from loss_func.get_loss import get_loss_func
from trainer.train_riga_unet_le import validate_riga_le


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, required=True, help='Path to model weights')
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--dataroot", type=str, default='../data/DiscRegion/DiscRegion')
    parser.add_argument("--dataset", choices=["RIGA"], default="RIGA")
    parser.add_argument("--rater_num", type=int, default=6)
    parser.add_argument("--loss_func", choices=["bce"], default="bce")
    parser.add_argument("--notes", type=str, default='')

    # architecture
    parser.add_argument("--net_arch", choices=["TAB", "Unet", "UnetLE"], default="Unet")
    parser.add_argument("--le_skip_encoder", action='store_true')
    parser.add_argument("--skip_head_eval", type=int, default=0)

    # details of dataset
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_channel", type=int, default=3)

    # training settings: classes; bs; lr; EPOCH; device_id
    parser.add_argument('--random_seed', type=int, default=27)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    
    # label settings
    parser.add_argument(
        '--gt_type_train', 
        type=int,
        choices=[-1, 0, 1, 2, 3, 4, 5], 
        default=-1)

    args = parser.parse_args()

    args.use_mix_label = False
    args.standardize = True
    args.validate = True
    print(args)

    return args


def get_model_outputs(args, model, val_set, loss_function, skip_idx=None, get_logits=False):
    model = model.cuda()
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True)

    model.eval()

    outputs_model = []
    masks = []
    for step, data in enumerate(val_loader):
        with torch.no_grad():
            imgs = data['image'].cuda()
            mask = data['mask']

            outputs = model({'image': imgs})

            outputs_sigmoid = []
            outputs_raw = []
            if skip_idx is not None:
                outputs = outputs[skip_idx:]

            for i, out in enumerate(outputs):
                out = torch.nn.functional.interpolate(out, size=mask[0].shape[2:])
                if get_logits:
                    outputs_raw.append(out)
                else:
                    outputs_sigmoid.append(torch.sigmoid(out))

        if get_logits:
            outputs_model.append(outputs_raw)
        else:
            outputs_model.append(outputs_sigmoid)
        masks.append(mask)
    return outputs_model, masks


def main():
    args = parse_args()
    model = build_model(args)
    train_set, valid_set, test_set = getDataset(args, validate=args.validate)
    model.load_state_dict(torch.load(args.path)['model'])

    out_path = os.path.dirname(os.path.dirname(args.path))
    out_path = f'{out_path}/metrics/u_metrics.txt'

    cup_idx = 1
    disc_idx = 0
    loss_func = get_loss_func(args)

    skip_idx = None if args.skip_head_eval == 0 else args.skip_head_eval 

    with open(out_path, '+a') as f:
        f.write(f'\n {datetime.datetime.now()}\n')
        f.write("Valid set results (loss, dice_disc, dice_cup): \n")
        f.write(str(validate_riga_le(args, model, valid_set, loss_func, skip_idx=skip_idx)))
        f.write('\n')

        f.write("Test set results: \n")
        f.write(str(validate_riga_le(args, model, test_set, loss_func, skip_idx=skip_idx)))
        f.write('\n\n')
    
    model_outputs, masks = get_model_outputs(args, model, test_set, loss_func, skip_idx=skip_idx)
    outputs_variance_cup = [torch.stack(out)[:, :, cup_idx, :, :].var(dim=0).sum().cpu().item() for out in model_outputs]
    outputs_variance_disc = [torch.stack(out)[:, :, disc_idx, :, :].var(dim=0).sum().cpu().item() for out in model_outputs]
    mask_variance_cup = [torch.stack(out)[:, :, cup_idx, :, :].var(dim=0).sum().cpu().item() for out in masks]
    mak_variance_disc = [torch.stack(out)[:, :, disc_idx, :, :].var(dim=0).sum().cpu().item() for out in masks]

    with open(out_path, '+a') as f:
        f.write("Spearmanr results: \n")
        f.write(f'Disc: {str(spearmanr(mak_variance_disc, outputs_variance_disc))}\n')
        f.write(f'Cup: {str(spearmanr(mask_variance_cup, outputs_variance_cup))}\n')
        f.write("Distance corr results: \n")
        f.write(f'Disc: {str(distance_correlation(mak_variance_disc, outputs_variance_disc))}\n')
        f.write(f'Cup: {str(distance_correlation(mask_variance_cup, outputs_variance_cup))}\n')

    model_outputs, masks = get_model_outputs(args, model, test_set, loss_func, skip_idx=skip_idx, get_logits=True)
    model_outputs_np = np.array([torch.stack(out).cpu().numpy() for out in model_outputs])
    labels = np.array([torch.stack(mask).mean(axis=0).cpu().numpy() for mask in masks])
    uncertainty_metrics = get_uncertainty_metrics(model_outputs_np[:, :, 0, :, :], labels[:, 0, :, :], T=0)
    
    with open(out_path, '+a') as f:
        for key in uncertainty_metrics[0].keys():
            f.write(
                f'{key}: mean: {np.mean(uncertainty_metrics[0][key])} std: {np.std(uncertainty_metrics[0][key])}\n'
            )


if __name__ == "__main__":
    main()