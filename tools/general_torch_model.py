# coding:utf-8
import torch
import numpy as np
import torch.nn as nn


class GeneralTorchModel(nn.Module):
    def __init__(self, model, n_class=10, im_mean=None, im_std=None,ismnist=False):
        super(GeneralTorchModel, self).__init__()
        self.model = model
        self.model.cuda().eval()
        self.num_queries = 0
        self.im_mean = im_mean
        self.im_std = im_std
        self.n_class = n_class
        self.ismnist = ismnist

    def forward(self, image):
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        if self.ismnist:
            if image.shape[1]==3:
                image_todo = image[:,0:1,:,:].clone()
        else:
            image_todo = image
        image_todo = self.preprocess(image_todo)
        logits = self.model(image_todo)
        return logits

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            processed = torch.from_numpy(image).type(torch.FloatTensor)
        else:
            processed = image

        if self.im_mean is not None and self.im_std is not None:
            im_mean = torch.tensor(self.im_mean).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            im_std = torch.tensor(self.im_std).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            processed = (processed - im_mean) / im_std
        return processed

    def predict_prob(self, image):
        with torch.no_grad():
            if len(image.size()) != 4:
                image = image.unsqueeze(0)
            if self.ismnist:
                if image.shape[1]==3:
                    image_todo = image[:,0:1,:,:].clone()
            else:
                image_todo = image
            image_todo = self.preprocess(image_todo)
            logits = self.model(image_todo)
            self.num_queries += image_todo.size(0)
        return logits

    def predict_label(self, image):
        image = image.cuda()
        logits = self.predict_prob(image)
        _, predict = torch.max(logits, 1)
        #predict = predict.cpu() 
        return predict