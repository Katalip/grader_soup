import torch
from utils.generate import generate_output_folder
from models.bulid import build_model
from loss_func.get_loss import get_loss_func
from dataset.get_dataset import getDataset

from trainer.train_riga import train_riga_tab, test_riga_tab
from trainer.train_riga_unet_le import train_riga_le, test_riga_le
from trainer.train_hecktor import train_hecktor, test_hecktor
from trainer.train_hecktor_unet_le import train_hecktor_le, test_hecktor_le


def train(args):
    log_folder, checkpoint_folder, visualization_folder, metrics_folder, gt_train_name = generate_output_folder(args)

    # network
    model = build_model(args)

    # load pretrained params
    if args.pretrained == 1:
        params = torch.load(args.pretrained_dir)
        model_params = params['model']
        model.load_state_dict(model_params)

    # dataset
    if args.dataset == 'Hecktor':
        dataset = getDataset(args, validate=args.validate)
    else:
        train_set, valid_set, test_set = getDataset(args, validate=args.validate)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # loss_func
    loss_func = get_loss_func(args)

    if args.dataset == "RIGA" and args.net_arch == 'Unet':
        train_riga_tab(args, log_folder, checkpoint_folder, visualization_folder, metrics_folder, model, optimizer,
                    loss_func, train_set, valid_set, test_set, gt_train_name)
    elif args.dataset == "RIGA" and args.net_arch == 'UnetLE':
        train_riga_le(args, log_folder, checkpoint_folder, visualization_folder, metrics_folder, model, optimizer,
                    loss_func, train_set, valid_set, test_set, gt_train_name)
    elif args.dataset == "Hecktor" and args.net_arch == 'Unet':
        train_hecktor(args, log_folder, checkpoint_folder, visualization_folder, metrics_folder, model, optimizer,
                    loss_func, dataset, gt_train_name)
    elif args.dataset == "Hecktor" and args.net_arch == 'UnetLE':
        train_hecktor_le(args, log_folder, checkpoint_folder, visualization_folder, metrics_folder, model, optimizer,
                    loss_func, dataset, gt_train_name)

    



def test(args):
    log_folder, checkpoint_folder, visualization_folder, metrics_folder, gt_train_name = generate_output_folder(args)

    # network
    model = build_model(args)

    # load pretrained params
    params = torch.load(checkpoint_folder + "/last.pt")
    model_params = params['model']
    model.load_state_dict(model_params)

    if args.dataset == 'Hecktor':
        test_set = getDataset(args, validate=args.validate)
    else:
        train_set, valid_set, test_set = getDataset(args, validate=args.validate)

    if args.dataset == "RIGA":
        test_riga_tab(args, visualization_folder, metrics_folder, model, test_set)
    elif args.dataset == "Hecktor" and args.net_arch == 'Unet':
        test_hecktor(args, visualization_folder, metrics_folder, model, test_set)
    elif args.dataset == "Hecktor" and args.net_arch == 'UnetLE':
        test_hecktor_le(args, visualization_folder, metrics_folder, model, test_set)
    
