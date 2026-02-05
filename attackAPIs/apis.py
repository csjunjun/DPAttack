import requests
import concurrent.futures
import io
from google.cloud import vision
import json,proto

import os
import torch
import numpy as np
import json
from torch import nn
import torchvision.models as models
from PIL import Image
import random
from datetime import datetime


import asyncio

from googletrans import Translator

from tencentcloud.common import credential
from tencentcloud.tiia.v20190529 import tiia_client, models
import base64,pickle
import clip

class Imagga:

    def __init__(self, api_key, api_secret, concurrency=1):
        self.api_key = api_key
        self.api_secret = api_secret
        self.concurrency = concurrency

        self.url = 'https://api.imagga.com/v2/tags'

    def predict(self, image_path):
        response = requests.post(
            self.url,
            auth=(self.api_key, self.api_secret),
            files={'image': open(image_path, 'rb')})

        return response.json()['result']['tags']


    def predictX(self, image_paths):
        y_preds = []
        y_index = []
        y_executors = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            for i, image_path in enumerate(image_paths):
                # Load the input image and construct the payload for the request
                y_executors[executor.submit(self.predict, image_path)] = i

            for y_executor in concurrent.futures.as_completed(y_executors):
                y_index.append(y_executors[y_executor])
                y_preds.append(y_executor.result())

            y_preds = [y for _, y in sorted(zip(y_index, y_preds))]

        return y_preds
    

#from google.protobuf.json_format import MessageToJson
#gcloud auth application-default set-quota-project
class CloudVision:
    def __init__(self, concurrency=1):
        self.concurrency = concurrency

    def predict(self, image_path):
        client = vision.ImageAnnotatorClient()
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = client.label_detection(image=image)
        json_str = proto.Message.to_json(response)
        json_str = json_str.replace("\n","")
        res_json = json.loads(json_str)
        return res_json

    def predictX(self, image_paths):
        y_preds = []
        y_index = []
        y_executors = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            for i, image_path in enumerate(image_paths):
                # Load the input image and construct the payload for the request
                y_executors[executor.submit(self.predict, image_path)] = i

            for y_executor in concurrent.futures.as_completed(y_executors):
                y_index.append(y_executors[y_executor])
                y_preds.append(y_executor.result())

            y_preds = [y for _, y in sorted(zip(y_index, y_preds))]

        return y_preds



class MyImagga():
    def __init__(self):

        api_key = '' # 
        api_secret =''

        self.imagga_client = Imagga(api_key, api_secret, concurrency=2)
        self.clean_image_tag = []

    def setClean(self,lista):
        self.clean_image_tag = lista
    
 
    
    def predict_label(self, image,return_res=False,request_res=None,ta="ADBA"):
        if request_res is None:
            timpstampnow = int(round(datetime.now().timestamp()))+random.randint(0,1000)
            savep = f"temp_{timpstampnow}_{ta}.png"
            Image.fromarray(np.uint8(np.round(image[0].permute(1,2,0).detach().cpu().numpy()*255))).save(savep)
            new_ress = self.imagga_client.predictX([savep])
            os.remove(savep)
        else:
            new_ress = request_res
        tag_list = []  
        ui = 0
        output_list = len(new_ress[ui])   
        
        for uj in range(output_list):

            tag = new_ress[ui][uj]['tag']['en']
            score = new_ress[ui][uj]['confidence']
            if score<100:
                if uj==0:
                    tag_list.append(tag)
                break
            else:
                tag_list.append(tag)
        if return_res:
            return tag_list,new_ress
        else:
            return tag_list

    
    def compare_label(self, ae_tag_list):
        att_succ_flag = True
        for ae_tag in ae_tag_list:
            if ae_tag in self.clean_image_tag:
                att_succ_flag = False
                break
        return att_succ_flag

class MyGoogle():
    def __init__(self):
        self.vision_client = CloudVision(concurrency=8)  

    def setClean(self,lista,score):
        self.clean_image_tag = lista
        self.clean_image_score = score

    def predict_label(self,oriimaggapath=None,respath=None):

        if not os.path.exists(respath):
            response = self.vision_client.predict(oriimaggapath)
            with open(respath, 'w') as f:
                json.dump(response,f)
        else:
            with open(respath, 'r') as f:
                response = json.load(f)
        if len(response["labelAnnotations"])>0:
            if "description" in response["labelAnnotations"][0].keys():
                google_label = response["labelAnnotations"][0]["description"]
            else:
                google_label = ""
            if "score" in response["labelAnnotations"][0].keys():
                google_score = response["labelAnnotations"][0]["score"]
            else:
                google_score = 0
        else:
            google_label = ""
            google_score = 0
        if google_label == "":
            print(f"google_label:blank")
        if google_score == 0:
            print(f"google_score:0")
        return google_label,google_score
    


async def translate_text(text):
    async with Translator() as translator:
        result = await translator.translate(text, dest='en',src='zh-CN')
        print(result)  # <Translated src=ko dest=en text=Good evening. pronunciation=Good evening.>
        return result

class BaiduAPI:
    '''
    API
    '''
    def __init__(self,savep=""):
        self.request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
        self.api_key = ""
        self.secret_key= ""
        self.get_token()
        self.outputp = savep

    def setgt(self,gtlabel):
        self.gtlabel = gtlabel      
    def ismatchHard(self,text1):
        if self.gtlabel.find(text1) != -1 or text1.find(self.gtlabel) != -1:
            return True
        print(f"compare: {self.gtlabel} and {text1}")
        return False


    def get_token(self):
            
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.api_key}&client_secret={self.secret_key}"
        
        payload = ""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        self.token = json.loads(response.text)["access_token"]
    
    def get_labels(self,imgpath):
        with open(imgpath, 'r') as fcc_file:
            tmp = json.load(fcc_file)
        if "result_num" not in tmp.keys():
            return ""

        if tmp['result_num']>0:
            result = tmp['result'] 
            for i in range(1):
                chinese_name = result[i]['keyword']
                english_name = asyncio.run(translate_text(chinese_name)).text
                score = result[i]['score']
                print(f"Chinese name: {chinese_name}, Englishe name: {english_name}, Confidence: {score}")
        else:
            english_name=""
        return english_name


    def detect_labels(self,imgpath,save=False):
 
        
        f = open(imgpath, 'rb')
        img = base64.b64encode(f.read())

        params = {"image":img}
        request_url = self.request_url + "?access_token=" + self.token 
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(request_url, data=params, headers=headers)
        tmp = json.loads(response.text)
        print(f"response: {tmp}")
        if "result_num" not in tmp.keys():
            return ""
        if tmp['result_num']>0:
            result = tmp['result'] 
            for i in range(1):
                chinese_name = result[i]['keyword']
                english_name = asyncio.run(translate_text(chinese_name)).text
                score = result[i]['score']
                print(f"Chinese name: {chinese_name}, Englishe name: {english_name}, Confidence: {score}")
        else:
            english_name=""
        if save:
            filename = imgpath.split('/')[-1].split('.')[0]
            outputp = os.path.join(self.outputp, filename+".json")
            if not os.path.exists(self.outputp):
                os.makedirs(self.outputp)
            with open(outputp, 'w') as f:
                f.write(json.dumps(tmp))
        return english_name


async def translate_text(text):
    async with Translator() as translator:
        result = await translator.translate(text, dest='en',src='zh-CN')
        print(result)  # <Translated src=ko dest=en text=Good evening. pronunciation=Good evening.>
        return result
        # result = await translator.translate('안녕하세요.', dest='ja')
        # print(result)  # <Translated src=ko dest=ja text=こんにちは。 pronunciation=Kon'nichiwa.>

        # result = await translator.translate('veritas lux mea', src='la')
        # print(result)  # <Translated src=la dest=en text=The truth is my light pronunciation=The truth is my light>

class TencentAPI():
    def __init__(self,outputp):
        self.cred = credential.Credential('', '')
        self.client = tiia_client.TiiaClient(self.cred, 'ap-shanghai')
        self.outputp = outputp
        self.clipmodel = clip.load("ViT-B/32", device="cuda:0", jit=False)[0]
    #  https://cloud.tencent.com/document/product/865/75196
    # def detect_labels(self,image_path,query,ground_name=""):
    def setgt(self,gtlabel):
        self.gtlabel = gtlabel.split(",")        
        with torch.no_grad():
            adv_vit_text_token    = clip.tokenize(self.gtlabel , truncate=True).cuda()
            adv_vit_text_features = self.clipmodel.encode_text(adv_vit_text_token)
            adv_vit_text_features = adv_vit_text_features / adv_vit_text_features.norm(dim=1, keepdim=True)
            adv_vit_text_features = adv_vit_text_features.detach()
        self.gtfeature = adv_vit_text_features

    def ismatch(self,text1list):
        with torch.no_grad():
            adv_vit_text_token    = clip.tokenize(text1list, truncate=True).cuda()
            adv_vit_text_features = self.clipmodel.encode_text(adv_vit_text_token)
            adv_vit_text_features = adv_vit_text_features / adv_vit_text_features.norm(dim=1, keepdim=True)
            adv_vit_text_features = adv_vit_text_features.detach()
        return torch.cosine_similarity(adv_vit_text_features, self.gtfeature.repeat(len(adv_vit_text_features),1),dim=1)
    def ismatchHard(self,text1list):
        for text in text1list:
            for gttext in self.gtlabel:
                if gttext.find(text) != -1 or text.find(gttext) != -1:
                    return True
        return False
    def detect_labels(self,image_path,query,ground_name="",save=True):

        with open(image_path, 'rb') as f:
            image_data = f.read()
        req = models.DetectLabelRequest()
        encoded_bytes = base64.b64encode(image_data)
        encoded_str = encoded_bytes.decode('utf-8')
        #req.Image = image_data
        req.ImageBase64=encoded_str
        resp = self.client.DetectLabelPro(req)
        if save:
            filename = image_path.split('/')[-1].split('.')[0]
            outputp = os.path.join(self.outputp, filename)
            if not os.path.exists(outputp):
                os.mkdir(outputp)
            with open(f"{outputp}/{query}.pickle", "wb") as out_file:
                pickle.dump(resp,out_file)

        ch_names,engnames = [],[]
        for datai in resp.Labels:
            name = datai.Name
            engname = asyncio.run(translate_text(name))
            ch_names.append(name)
            engnames.append(engname.text)  
  
        return ch_names,engnames,resp.Labels

    def get_labels(self,image_path,ground_name=""):
        with open(image_path, "rb") as out_file:
            data = pickle.load(out_file)
        ch_names,engnames = [],[]
        for datai in data.Labels:
            name = datai.Name
            engname = asyncio.run(translate_text(name))
            ch_names.append(name)
            engnames.append(engname.text)
        
        return ch_names,engnames,data.Labels
    # with open(f"../tencent_response/ILSVRC2012_val_00000001/1.pickle", "rb") as out_file:
    #     data = pickle.load(out_file)
    # def getEnLabels(self, image_path):
    #     with open(image_path, "rb") as out_file:
    #         data = pickle.load(out_file)
    #     ch_names,engnames = [],[]
    #     for datai in data.Labels:
    #         name = datai.Name
    #         engname = asyncio.run(translate_text(name))
    #         ch_names.append(name)
    #         engnames.append(engname.text)
    #     return ch_names,engnames

# example usage:
# labels = detect_labels('../ImageNet/val/ILSVRC2012_val_00000001.JPEG')
# print("Detected Tags:", [label.Name for label in labels])

if __name__ == "__main__":

    
    client = vision.ImageAnnotatorClient()
    print("auth sucess")

