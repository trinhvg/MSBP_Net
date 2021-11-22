import importlib


def define_network(network_name, nr_class):
    " Return a network by its name"
    if network_name == "Resnet":
        net_def = importlib.import_module('model.resnet')  # dynamic import
        net = net_def.resnet50(pretrained=True, num_classes=nr_class)
    elif network_name == 'VGG':
        print('self.exp_mode == VGG')
        net_def = importlib.import_module('model.vgg16')  # dynamic import
        net = net_def.vgg16_bn(nr_classes=nr_class, pretrained=True)
    elif network_name == "MobileNetV1":
        net_def = importlib.import_module('model.mobilenetv1')  # dynamic import
        net = net_def.MobileNet(num_classes=nr_class)
    elif network_name == "EfficientNet":
        net_def = importlib.import_module('model.efficientnet.model')  # dynamic import
        net = net_def.efficientnet(model_name='efficientnet-b1', pretrained=True, num_classes=nr_class)
    elif network_name == "ResNeSt":
        net_def = importlib.import_module('model.resnest.torch.resnest')  # dynamic import
        net = net_def.resnest50(pretrained=True, num_classes=nr_class)
    elif network_name == "MuDeep":
        net_def = importlib.import_module('prenet.mudeep')
        net = net_def.MuDeep(num_classes=nr_class, fc_in=16)
    elif network_name == "MSDNet":
        net_def = importlib.import_module('prenet.MSDNet')
        net = net_def.msdnet(pretrained=True, nr_class=nr_class)
    elif network_name == "Res2Net":
        net_def = importlib.import_module('prenet.res2net')
        net = net_def.res2net50(pretrained=True, num_classes=nr_class)
    else:
        net_def = importlib.import_module('ResNet.code.Resnet_MSBP')  # dynamic import
        net = net_def.resnet_msbp(exp_mode=network_name, nr_classes=nr_class)
    return net
