import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

#from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def get_resnet101(num_classes, pretrained):
    model = torchvision.models.resnet101(pretrained=pretrained)
    model.fc = torch.nn.Linear(2048, num_classes)
    return model


def get_resnet50(num_classes, pretrained):
    model = torchvision.models.resnet50(pretrained=pretrained)
    model.fc = torch.nn.Linear(2048, num_classes)
    return model


def get_resnet18(num_classes, pretrained):
    model = torchvision.models.resnet18(pretrained=True)
    #ckp = torch.load('/home/work/cache/torch/hub/checkpoints/resnet18-5c106cde.pth')
    #model.load_state_dict(ckp)
    model.fc = torch.nn.Linear(512, num_classes)
    return model


def get_resnet34(num_classes, pretrained):
    model = torchvision.models.resnet34(pretrained=pretrained)
    model.fc = torch.nn.Linear(512, num_classes)
    return model

def get_densenet121(num_classes, pretrained):
    model = torchvision.models.densenet161(pretrained=pretrained)
    model.classifier = torch.nn.Linear(2208, num_classes)
    return model


if __name__ == "__main__":
    a = 2
