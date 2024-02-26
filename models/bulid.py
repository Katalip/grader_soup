from models.TAB import build_TAB
from models.unet import UNet
from models.unet_le import UNetLE


def build_model(args):
    """
    return models
    """
    if args.net_arch == "TAB":
        model = build_TAB(args)
    elif args.net_arch == 'Unet':
        input_channels = 2 if args.dataset == 'Hecktor' else 3
        model = UNet(input_channels=input_channels)
        print('unet_selected')
    elif args.net_arch == 'UnetLE':
        input_channels = 2 if args.dataset == 'Hecktor' else 3
        model = UNetLE(skip_encoder=args.le_skip_encoder, input_channels=input_channels)
        print('unet_le_selected')
    return model

