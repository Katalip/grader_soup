import torch
from dataset.DiscRegion import get_data_path_list, Disc_Cup
from dataset.hecktor import Hecktor22


def getDataset(args, validate=False, transforms=None):

    if args.dataset == "RIGA":
        train_set_path_list = get_data_path_list(args.dataroot, ['BinRushed', 'MESSIDOR'], 6)
        test_set_path_list = get_data_path_list(args.dataroot, ['Magrabia'], 6)  # rater_num==6
        train_set = Disc_Cup(args, train_set_path_list, augs=transforms)
        test_set = Disc_Cup(args, test_set_path_list, augs=transforms)

        if validate:  # True
            train_size = int(train_set.__len__()*0.8)
            fixed_generator = torch.Generator().manual_seed(args.random_seed)
            train_dataset, valid_dataset = torch.utils.data.random_split(
                train_set, 
                [train_size, train_set.__len__()-train_size],
                generator=fixed_generator)
            return train_dataset, valid_dataset, test_set
        else:  # False
            return train_set, None, test_set
    elif args.dataset == "Hecktor":
        dataset = Hecktor22(args)
        return dataset
