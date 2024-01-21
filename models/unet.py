import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch import Unet


class UNet(nn.Module):
    """Implements U-Net architecture with custom backbone

    Attributes:
        encoder_name (str, optional): encoder architecture and size
        encoder_pretrain (str, optional): Choice of pretrained weights. Additional options are available in
                         segmentation models pytorch package documentation. Defaults to 'imagenet'.
        output_type (set, optional): model outputs
        n_classes (int, optional): N of classes being predicted. Defaults to 1.
    """

    def load_pretrain(self):
        pretain = self.encoder_pretrain
        print("load %s" % pretain)
        state_dict = torch.load(
            pretain, map_location=lambda storage, loc: storage
        )  # True
        print(self.encoder.load_state_dict(state_dict, strict=False))  # True

    def __init__(
        self,
        encoder_name = 'resnet50',
        encoder_pretrain="imagenet",
        output_type={'raw'},
        n_classes=2
    ):
        """
        Args:
            encoder_name (str, optional): encoder architecture and size
            encoder_pretrain (str, optional): Choice of pretrained weights. Additional options are available in
                         segmentation models pytorch package documentation. Defaults to 'imagenet'.
            output_type (set, optional): model outputs
            n_classes (int, optional): N of classes being predicted. Defaults to 1.
        """
        super(UNet, self).__init__()
        decoder_dim = [256, 128, 64, 32, 16]
        self.n_classes = n_classes

        self.output_type = output_type
        self.encoder_pretrain = encoder_pretrain

        model = Unet(encoder_name=encoder_name, encoder_weights=encoder_pretrain)
        self.encoder = model.encoder
        self.decoder = model.decoder

        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dim[-1], n_classes, kernel_size=1),
        )
        

    def forward(self, batch):
        x = batch["image"]
        encoder = self.encoder(x)
        last = self.decoder(*encoder)
        logit = self.logit(last)

        output = {}
        output["raw"] = logit

        if "loss" in self.output_type:
            if self.n_classes == 1:
                output["bce_loss"] = F.binary_cross_entropy_with_logits(
                    logit, batch["mask"]
                )

        if "inference" in self.output_type:
            probability_from_logit = torch.sigmoid(logit)
            output["probability"] = probability_from_logit

        return output


def test_network(pretrained=False):
    batch_size = 2
    image_size = 768

    # ---
    batch = {
        "image": torch.from_numpy(
            np.random.uniform(-1, 1, (batch_size, 3, image_size, image_size))
        ).float(),
        "mask": torch.from_numpy(
            np.random.choice(2, (batch_size, 1, image_size, image_size))
        ).float(),
    }
    batch = {k: v.cuda() for k, v in batch.items()}

    net = UNet().cuda()

    if pretrained:
        net.load_pretrain()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            output = net(batch)

    print("batch")
    for k, v in batch.items():
        print("%32s :" % k, v.shape)

    print("output")
    for k, v in output.items():
        if "loss" not in k:
            print("%32s :" % k, v.shape)
    for k, v in output.items():
        if "loss" in k:
            print("%32s :" % k, v.item())


if __name__ == "__main__":
    test_network()