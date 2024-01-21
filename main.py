import argparse
import numpy as np
from run import train, test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--dataroot", type=str, default='/.../datasets/DiscRegion')
    parser.add_argument("--dataset", choices=["RIGA"], default="RIGA")
    parser.add_argument("--phase", choices=["train", "test"], default="train")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--net_arch", choices=["TAB", 'Unet'], default="Unet")
    parser.add_argument("--rater_num", type=int, default=6)
    parser.add_argument("--loss_func", choices=["bce"], default="bce")

    # pretrained params
    parser.add_argument("--pretrained", type=int, default=0, help="whether to load pretrained models.")
    parser.add_argument("--pretrained_dir", type=str, default="none", help="the path of pretrained models.")

    # details of dataset
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_channel", type=int, default=3)
    parser.add_argument("--standardize", action="store_true")


    # training settings: classes; bs; lr; EPOCH; device_id
    parser.add_argument('--random_seed', type=int, default=27)
    parser.add_argument('--checkpoint_frequency', type=int, default=20)
    parser.add_argument("--rank", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--num_epoch",  default=200, type=int)
    parser.add_argument("--device_id", default="0", choices=["0", "1", "2", "3", "4", "5", "6", "7"], help="gpu ID.")
    parser.add_argument("--loop", default=0, type=int, help="this is the {loop}-th run.")
    # label settings
    parser.add_argument(
        '--gt_type_train', 
        type=int,
        choices=[-1, 0, 1, 2, 3, 4, 5], 
        default=-1)
    # mix label settings
    parser.add_argument("--use_mix_label", action="store_true")
    parser.add_argument(
        "--mix_label_type", choices=["intersection", "union"], type=str, 
        default='intersection')
    parser.add_argument('--gt_index_1', type=int, default=0)
    parser.add_argument('--gt_index_2', type=int, default=0)

    args = parser.parse_args()
    print(args)

    if args.phase == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
