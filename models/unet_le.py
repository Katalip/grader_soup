import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch import Unet


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=None, activation=None):
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        super().__init__(dropout, conv2d)


class UNetLE(nn.Module):
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
        super(UNetLE, self).__init__()
        decoder_dims = [256, 128, 64, 32, 16]
        self.n_classes = n_classes

        self.output_type = output_type
        self.encoder_pretrain = encoder_pretrain

        model = Unet(encoder_name=encoder_name, encoder_weights=encoder_pretrain)
        self.encoder = model.encoder
        self.decoder = model.decoder

        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dims[-1], n_classes, kernel_size=1),
        )

        encoder_dims = self.encoder.out_channels
        self.encoder_output_heads = nn.ModuleList([
                SegmentationHead(
                    in_channels=in_channel,
                    out_channels=self.n_classes,
                    kernel_size=1,
                ) for in_channel in encoder_dims[1:]
            ])
        
        self.decoder_output_heads = nn.ModuleList([
                SegmentationHead(
                    in_channels=in_channel,
                    out_channels=self.n_classes,
                    kernel_size=1,
                ) for in_channel in decoder_dims
            ])

        
    def forward(self, batch):
        x = batch["image"]
        encoder = self.encoder(x)
        encoder = encoder[1:]
        outputs = [
            self.encoder_output_heads[i](encoder[i]) for i in range(len(encoder))
        ]

        encoder = encoder[::-1]  # reverse channels to start from head of encoder

        head = encoder[0]
        skips = encoder[1:]

        x = self.decoder.center(head)
        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            seg_head_output = self.decoder_output_heads[i](x)
            outputs.append(seg_head_output)

        return outputs


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

    net = UNetLE().cuda()

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