from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half
from .convnextv2 import convnextv2_atto, convnextv2_femto, convnextv2_pico, convnextv2_nano, convnextv2_tiny, convnextv2_small
from .poolformerv2 import poolformerv2_s12, poolformerv2_s24, poolformerv2_s36
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'convnextv2_atto': convnextv2_atto,
    'convnextv2_femto': convnextv2_femto,
    'convnextv2_pico': convnextv2_pico,
    'convnextv2_nano': convnextv2_nano,
    'convnextv2_tiny': convnextv2_tiny,
    'convnextv2_small': convnextv2_small,
    'poolformerv2_s12': poolformerv2_s12,
    'poolformerv2_s24': poolformerv2_s24,
    'poolformerv2_s36': poolformerv2_s36,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
}
