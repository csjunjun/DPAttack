import torch
import open_clip
import numpy as np
from torchvision import transforms


class MyCliptModel(torch.nn.Module):
    def __init__(self,model,clipM,clipdataset,device) -> None:
        super().__init__()
        self.model = model
        self.basepp = "../data/objectNet/objectnet-1.0/mappings"
       
        with open(f"{self.basepp}/cliptest1000_withoutimagenetclasswithidx.txt","r") as f:
            self.imagepathandlabels = f.read().splitlines()
            
        tokenizer = open_clip.get_tokenizer(clipM)
        print(f"{clipM} model pretrained on {clipdataset} dataset loaded.")                    
        self.classlist = np.load(f"{self.basepp}/objectNetClasslistWithoutImagenet.npy")
        text = tokenizer(self.classlist).to(device)
        self.text_features = self.model.encode_text(text)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        self.transform_clip =  transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])  
    def forward(self,x):
        with torch.no_grad():
        
            output_inter1_proc = self.transform_clip(x)
            image_features = self.model.encode_image(output_inter1_proc)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        return probs
    