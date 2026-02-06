
from torchvision import models
import torch,sys
from .general_torch_model import GeneralTorchModel
import json,os

class MyRobustModel(torch.nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model
    def forward(self,x):
        return self.model(x)[0]
def fetchImageNetModels(modelname):
    if  modelname == 'resnet50':
        
        model = models.resnet50(pretrained=True)
        model = torch.nn.DataParallel(model)
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])

    elif modelname == "DeiTB":
        from robustbench import load_model #no gdrive id
        net = load_model(model_name="Tian2022Deeper_DeiT-B", dataset="imagenet",threat_model="corruptions",model_dir="/data/code/attacks/ADBA/code/models").cuda()
        torch_model = GeneralTorchModel(net, n_class=1000, im_mean=None,
                                        im_std=None)

    elif modelname == 'regnet':
        model = models.regnet_x_8gf(pretrained=True)
        model = torch.nn.DataParallel(model)
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif modelname == "NoisyMix":
        from robustbench import load_model 
        net = load_model(model_name="Erichson2022NoisyMix", dataset="imagenet",threat_model="corruptions",model_dir="../models").cuda()
        torch_model = GeneralTorchModel(net, n_class=1000, im_mean=None,
                                        im_std=None)
    elif modelname == "HMany":

        _original_torch_load = torch.load

        def _unsafe_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)
        torch.load = _unsafe_torch_load
        from robustbench import load_model 
        net = load_model(model_name="Hendrycks2020Many", dataset="imagenet",threat_model="corruptions").cuda()
        torch_model = GeneralTorchModel(net, n_class=1000, im_mean=None,
                                        im_std=None)
    
    elif modelname == 'inv3':
        model = models.inception_v3(pretrained=True)
        model = torch.nn.DataParallel(model).cuda()
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif modelname == 'vit':
        model = models.vit_b_32(pretrained=True)
        model = torch.nn.DataParallel(model)
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif modelname == 'efficient':
        #model = models.__dict__["efficientnet_b0"]().cuda()
        # weight = torchvision.models.efficientnet_b0(models.EfficientNet_B0_Weights.DEFAULT)
        #weight = torch.load("model/efficientnet_b0_rwightman-7f5810bc.pth")
        #model.load_state_dict(weight)

        # model = models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # model = torch.nn.DataParallel(model)
        # torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
        #                                 im_std=[0.229, 0.224, 0.225])

        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        from torchvision.models._api import WeightsEnum
        from torch.hub import load_state_dict_from_url

        def get_state_dict(self, *args, **kwargs):
            kwargs.pop("check_hash")
            return load_state_dict_from_url(self.url, *args, **kwargs)
        WeightsEnum.get_state_dict = get_state_dict

        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])

    elif modelname == 'wrs50':
        from robustness import model_utils
        from robustness import datasets as datasetsr
        modelpath = '../microsoft_robust_models/wide_resnet50_2_linf_eps8.0_imgnet.ckpt'

        print(modelpath)
        net ,_ = model_utils.make_and_restore_model(arch="wide_resnet50_2",dataset=datasetsr.ImageNet(''), resume_path=modelpath, pytorch_pretrained=None,add_custom_forward=False)
        net = MyRobustModel(net)
        torch_model = GeneralTorchModel(net, n_class=1000, im_mean=None,
                                        im_std=None)
           
    elif modelname == "dense121":
        from mmpretrain import get_model 
        net = get_model('densenet121_3rdparty_in1k', pretrained=True)
        torch_model = GeneralTorchModel(net, n_class=1000, im_mean=None,
                                        im_std=None)   
    elif modelname == "SwinV2T":      
        model = models.swin_v2_t(weights="IMAGENET1K_V1")
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif modelname == "ConvNextBase":
        model = models.convnext_base(weights="IMAGENET1K_V1")
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])

    else:
        print("Invalid dataset")
        exit(1)
    return torch_model


def fetchPMNIST(modelname):
    if modelname=="Net28":
        from medmnist import INFO
        data_flag = 'pathmnist'

        info = INFO[data_flag]

        from models.pmnist.model import Net28
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        net = Net28(in_channels=n_channels, num_classes=n_classes)#80.7% acc
        net.load_state_dict(torch.load('models/pmnist/pathmnist.pth'))    
        net = net.cuda()
        net.eval()
        torch_model = GeneralTorchModel(net, n_class=n_classes, im_mean=[0.5,0.5,0.5],
                                        im_std=[0.5,0.5,0.5])
    return torch_model

def featchCifar10Models(modelname):
    from pathlib import Path

    root_path = Path(__file__).resolve().parent.parent

    sys.path.append(str(root_path))
    if modelname == 'res18':
        from models import CifarResnetModel
        target_model = CifarResnetModel.ResNet18()
        cp = torch.load('models/ckpt_resnet18_400e.pth')

        target_model= torch.nn.DataParallel(target_model).cuda()
        target_model.load_state_dict(cp['net'])
        target_model.eval()
        torch_model = GeneralTorchModel(target_model, n_class=10, im_mean=None, #ACC:92.16
                                        im_std=None)      
     
 
    elif modelname == "wrn":
       

        from robustbench import load_model #https://drive.google.com/uc?id=1-AaTrYt23WJFR22hXgBd-i6kjpsz6Hf2
        net = load_model(model_name="Cui2023Decoupled_WRN-28-10", dataset="cifar10",threat_model="Linf",model_dir="" \
        "yourmodelpath").cuda()
        
        torch_model = GeneralTorchModel(net, n_class=10, im_mean=None, #ACC:92.16
                                        im_std=None)  
   
    elif modelname== "vgg":

        TRAINED_MODEL_PATH = 'models/vgg16cifar10'
        filename = 'vgg16_cifar10.pth'
        from models.vgg16cifar10 import vgg
        with open(os.path.join(TRAINED_MODEL_PATH, 'config.json')) as fr:
            pretrained_model = vgg.Network(json.load(fr)['model_config'])
            pretrained_model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename))['state_dict'])
    
        pretrained_model = pretrained_model.cuda()
        pretrained_model.eval()
       
        torch_model = GeneralTorchModel(pretrained_model, n_class=10, im_mean=[0.4914, 0.4822, 0.4465], #ACC:92.16
                                        im_std=[0.2470, 0.2435, 0.2616])   

    else:
        print("Invalid victimmodel")
        exit(1)

    return torch_model


class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        # print (size)
        x = input.clone()
        for i in range(size[1]):
            x[:,i] = (x[:,i] - self.mean[i])/self.std[i]

        return x


def load_adv_imagenet(model_list = ['VGG16', 'Resnet18', 'Googlenet'],norm=True,type=""):
    nets = []

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for model_name in model_list:
        print(model_name)
        if  model_name =="Resnet50":
            pretrained_model = models.resnet50(pretrained=True)
       
        elif model_name == "ConvNextBase":
            pretrained_model = models.convnext_base(weights="IMAGENET1K_V1")
        elif model_name == "EfficientB3":
            from torchvision.models._api import WeightsEnum
            WeightsEnum.get_state_dict = get_state_dict 
            pretrained_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        elif model_name == "efficient":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            from torchvision.models._api import WeightsEnum
            from torch.hub import load_state_dict_from_url

            def get_state_dict(self, *args, **kwargs):
                kwargs.pop("check_hash")
                return load_state_dict_from_url(self.url, *args, **kwargs)
            WeightsEnum.get_state_dict = get_state_dict

            pretrained_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        elif model_name == "SwinV2T":
            pretrained_model = models.swin_v2_t(weights="IMAGENET1K_V1")
        elif model_name == "Densenet121":
            from mmpretrain import get_model 
            pretrained_model = get_model('densenet121_3rdparty_in1k', pretrained=True) 


        elif model_name =="wrs50":
            from robustness import model_utils
            from robustness import datasets as datasetsr
            modelpath = '../data/microsoft_robust_models/wide_resnet50_2_linf_eps8.0_imgnet.ckpt'
            
            print(modelpath)
            net ,_ = model_utils.make_and_restore_model(arch="wide_resnet50_2",dataset=datasetsr.ImageNet(''), resume_path=modelpath, pytorch_pretrained=None,add_custom_forward=False)
            net = MyRobustModel(net)
        elif model_name=="vit": 
            norm = False
            net = models.vit_b_32(pretrained=True)
            net = torch.nn.DataParallel(net)
            net = GeneralTorchModel(net, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                            im_std=[0.229, 0.224, 0.225])
        elif model_name=="inv3":
            norm = False
            net = models.inception_v3(pretrained=True)
            net = torch.nn.DataParallel(net)
            net = GeneralTorchModel(net, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                            im_std=[0.229, 0.224, 0.225])
        
        else:
            print(f"model not found:{model_name}")
            return
        if  model_name !="wrs50" and model_name !="vit" and model_name !="inv3" :
            if norm:
                net = nn.Sequential(
                    Normalize(mean, std),
                    pretrained_model
                )
            else:
                net = pretrained_model
        nets.append(net)

    for i in range(len(nets)):
        nets[i] = nets[i].cuda()
        nets[i].eval()
        
    return nets