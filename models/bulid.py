from models.TAB import build_TAB
from models.unet import UNet


def build_model(args):
    """
    return models
    """
    if args.net_arch == "TAB":
        model = build_TAB(args)
    elif args.net_arch == 'Unet':
        model = UNet()
        print('unet_selected')
    return model

