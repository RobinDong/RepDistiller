import timm
import torch.nn as nn


__all__ = [
    'convnextv2_atto', 'convnextv2_femto', 'convnextv2_pico',
    'convnextv2_nano', 'convnextv2_tiny', 'convnextv2_small',
]


class ConvNeXtV2Wrapper(nn.Module):
    def __init__(self, subname: str, num_classes=100):
        super().__init__()
        self.backbone = timm.create_model(
            f"convnextv2_{subname}.fcmae_ft_in1k",
            num_classes = num_classes,
            patch_size = 1,
        )
        self.backbone.stem[0].kernel_size = 4

    def forward(self, inp, is_feat=False, preact=False):
        features = []
        x = self.backbone.stem(inp)

        for index, stage in enumerate(self.backbone.stages):
            x = stage(x)
            features.append(x)

        out = self.backbone.norm_pre(x)
        out = self.backbone.forward_head(out)

        if is_feat:
            return features, out
        else:
            return out


def convnextv2_atto(**kwargs):
    return ConvNeXtV2Wrapper("atto", **kwargs)


def convnextv2_femto(**kwargs):
    return ConvNeXtV2Wrapper("femto", **kwargs)


def convnextv2_pico(**kwargs):
    return ConvNeXtV2Wrapper("pico", **kwargs)


def convnextv2_nano(**kwargs):
    return ConvNeXtV2Wrapper("nano", **kwargs)


def convnextv2_tiny(**kwargs):
    return ConvNeXtV2Wrapper("tiny", **kwargs)


def convnextv2_small(**kwargs):
    return ConvNeXtV2Wrapper("small", **kwargs)
