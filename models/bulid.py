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
        model = UNet()
        print('unet_selected')
    elif args.net_arch == 'UnetLE':
        model = UNetLE()
        print('unet_le_selected')
    return model

