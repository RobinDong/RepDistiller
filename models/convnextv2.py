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
        x = self.backbone.stem(inp)

        stages = self.backbone.stages
        f0 = x
        x = stages[0](x)
        f1 = x
        x = stages[1](x)
        f2 = x
        x = stages[2](x)
        f3 = x
        x = stages[3](x)

        out = self.backbone.norm_pre(x)
        f4 = x
        out = self.backbone.forward_head(out)

        if is_feat:
            return [f0, f1, f2, f3, f4], out
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
