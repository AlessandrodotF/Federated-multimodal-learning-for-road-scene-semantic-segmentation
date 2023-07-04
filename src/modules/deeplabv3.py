import copy
from collections import OrderedDict

import torch
from .mobilenetv2 import MobileNetV2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, DeepLabHead
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torch import nn
from torch.nn import functional as F


class MultiDeepLabV3(nn.Module):
    def __init__(self, backbone1, backbone2, classifier):
        super(MultiDeepLabV3, self).__init__()
        self.rgb_backbone = backbone1
        self.hha_backbone = backbone2
        self.classifier = classifier

    def forward(self, x_rgb=None, z_hha=None):
        if x_rgb is not None and z_hha is None:
            features = self.rgb_backbone(x_rgb)
            output = self.classifier(features["out"])

        elif x_rgb is None and z_hha is not None:
            features = self.hha_backbone(z_hha)
            output = self.classifier(features["out"])

        elif x_rgb is not None and z_hha is not None:
            features_rgb = self.rgb_backbone(x_rgb)
            features_hha = self.hha_backbone(z_hha)
            mean_features = {}
            for key in features_rgb.keys():
                mean_features[key] = (features_rgb[key] + features_hha[key]) / 2.0

            output = self.classifier(mean_features["out"])
            # x_rgb e z_hha hanno stesse dimensioni, prendo una delle due a caso
            output = F.interpolate(output, size=x_rgb.shape[-2:], mode='bilinear', align_corners=False)
        else:
            raise ValueError("Either x_rgb or z_hha should be provided.")
        # LADD/lib/python3.9/site-packages/torchvision/models/segmentation/_utils.py contiene "def forward" che ha un
        # interpolate che aggiusta le dimensioni, anche il caso base aveva il "problema" delle dimensioni.
        # (8,19,4,4) viene trasformato nella versione corretta (8,19,100,100)
        return output


def _multi_deeplabv3_mobilenetv2(backbone1: MobileNetV2, backbone2: MobileNetV2, num_classes: int) -> MultiDeepLabV3:
    backbone1 = backbone1.features
    backbone2 = backbone2.features

    out_pos = len(backbone1) - 1
    out_inplanes = backbone1[out_pos][0].out_channels

    return_layers = {str(out_pos): "out"}

    backbone1 = create_feature_extractor(backbone1, return_layers)
    backbone2 = create_feature_extractor(backbone2, return_layers)

    classifier = DeepLabHead(out_inplanes, num_classes)

    return MultiDeepLabV3(backbone1, backbone2, classifier)


def multi_deeplabv3_mobilenetv2(num_classes: int = 21, in_channels: int = 3) -> MultiDeepLabV3:
    width_mult = 1
    rgb_backbone = MobileNetV2(width_mult=width_mult, in_channels=in_channels)
    hha_backbone = MobileNetV2(width_mult=width_mult, in_channels=in_channels)

    model_urls = {
        0.5: 'https://github.com/d-li14/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_0.5-eaa6f9ad.pth',
        1.0: 'https://github.com/d-li14/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_1.0-0c6065bc.pth'
    }

    state_dict_rgb = load_state_dict_from_url(model_urls[width_mult], progress=True)
    state_dict_hha = load_state_dict_from_url(model_urls[width_mult], progress=True)

    state_dict_updated_rgb = state_dict_rgb.copy()
    state_dict_updated_hha = state_dict_hha.copy()


    for k, v in state_dict_rgb.items():
        if 'features' not in k and 'classifier' not in k:
            state_dict_updated_rgb[k.replace('conv', 'features.18')] = v
            del state_dict_updated_rgb[k]

    for k, v in state_dict_hha.items():
        if 'features' not in k and 'classifier' not in k:
            state_dict_updated_hha[k.replace('conv', 'features.18')] = v
            del state_dict_updated_hha[k]

    if in_channels == 4:
        aux_rgb = torch.zeros((32, 4, 3, 3))
        aux_hha = torch.zeros((32, 4, 3, 3))
        aux_rgb[:, 0, :, :] = copy.deepcopy(state_dict_updated_rgb['features.0.0.weight'][:, 0, :, :])
        aux_rgb[:, 1, :, :] = copy.deepcopy(state_dict_updated_rgb['features.0.0.weight'][:, 1, :, :])
        aux_rgb[:, 2, :, :] = copy.deepcopy(state_dict_updated_rgb['features.0.0.weight'][:, 2, :, :])
        aux_rgb[:, 3, :, :] = copy.deepcopy(state_dict_updated_rgb['features.0.0.weight'][:, 0, :, :])
        state_dict_updated_rgb['features.0.0.weight'] = aux_rgb

        aux_hha[:, 0, :, :] = copy.deepcopy(state_dict_updated_hha['features.0.0.weight'][:, 0, :, :])
        aux_hha[:, 1, :, :] = copy.deepcopy(state_dict_updated_hha['features.0.0.weight'][:, 1, :, :])
        aux_hha[:, 2, :, :] = copy.deepcopy(state_dict_updated_hha['features.0.0.weight'][:, 2, :, :])
        aux_hha[:, 3, :, :] = copy.deepcopy(state_dict_updated_hha['features.0.0.weight'][:, 0, :, :])
        state_dict_updated_hha['features.0.0.weight'] = aux_hha

    rgb_backbone.load_state_dict(state_dict_updated_rgb)
    hha_backbone.load_state_dict(state_dict_updated_hha)

    model = _multi_deeplabv3_mobilenetv2(rgb_backbone, hha_backbone, num_classes)
    model.task = 'segmentation'

    return model




def _deeplabv3_mobilenetv2(
        backbone: MobileNetV2,
        num_classes: int,
) -> DeepLabV3:
    backbone = backbone.features

    out_pos = len(backbone) - 1
    out_inplanes = backbone[out_pos][0].out_channels
    return_layers = {str(out_pos): "out"}

    backbone = create_feature_extractor(backbone, return_layers)
    classifier = DeepLabHead(out_inplanes, num_classes)
    #encoder=backbone,decoder=classifier
    return DeepLabV3(backbone, classifier)

def deeplabv3_mobilenetv2(
        num_classes: int = 21,
        in_channels: int = 3
) -> DeepLabV3:

    width_mult = 1
    backbone = MobileNetV2(width_mult=width_mult, in_channels=in_channels)
    model_urls = {
        0.5: 'https://github.com/d-li14/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_0.5-eaa6f9ad.pth',
        1.: 'https://github.com/d-li14/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_1.0-0c6065bc.pth'}
    state_dict = load_state_dict_from_url(model_urls[width_mult], progress=True)
    state_dict_updated = state_dict.copy()

    for k, v in state_dict.items():
        if 'features' not in k and 'classifier' not in k:
            state_dict_updated[k.replace('conv', 'features.18')] = v
            del state_dict_updated[k]

    if in_channels == 4:
        aux = torch.zeros((32, 4, 3, 3))
        aux[:, 0, :, :] = copy.deepcopy(state_dict_updated['features.0.0.weight'][:, 0, :, :])
        aux[:, 1, :, :] = copy.deepcopy(state_dict_updated['features.0.0.weight'][:, 1, :, :])
        aux[:, 2, :, :] = copy.deepcopy(state_dict_updated['features.0.0.weight'][:, 2, :, :])
        aux[:, 3, :, :] = copy.deepcopy(state_dict_updated['features.0.0.weight'][:, 2, :, :])
        state_dict_updated['features.0.0.weight'] = aux
    backbone.load_state_dict(state_dict_updated, strict=False)


    model = _deeplabv3_mobilenetv2(backbone, num_classes)
    model.task = 'segmentation'

    return model

