# coding:utf-8
import os
import subprocess,torch
import random
from PIL import Image
from torchvision import utils
from scipy.integrate import quad



def save_and_resize_image(image_tensor, file_path):# save adverserial examples
    utils.save_image(image_tensor, file_path)
    img = Image.open(file_path)
    img = img.resize((320, 320), Image.NEAREST)
    img.save(file_path, format='BMP')  


def concatenate_images(image_paths):# save adverserial examples
    images = [Image.open(x) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return new_img

""" """
def save_images(original_image, adversarial_image, adversarial_v, plustring,candi):# save adverserial examples
    folder_name = "adversarial_samples_ADBAV2ours_resnet50"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        

    original_file = os.path.join(folder_name, f"ORIG_{plustring}.bmp")
    adversarial_v_file = os.path.join(folder_name, f"ADVV_{plustring}.bmp")
    adversarial_file = os.path.join(folder_name, f"ADVO_{plustring}.bmp")
    combined_file = os.path.join(folder_name, f"COMB_{plustring}.bmp")
    init_file = os.path.join(folder_name, f"INIT_{plustring}.bmp")
    initdiff_file = os.path.join(folder_name, f"INITDIFF_{plustring}.bmp")
    print(f"combined_file save to {combined_file}")
    # Save and resize images in BMP format
    save_and_resize_image(original_image, original_file)
    save_and_resize_image(adversarial_image, adversarial_file)

    # Save perturbation image in BMP format
    # adversarial_v = adversarial_image - original_image
    save_and_resize_image(adversarial_v, adversarial_v_file)

    # Concatenate and save combined image in BMP format
    combined_image = concatenate_images([adversarial_v_file, original_file, adversarial_file])
    combined_image.save(combined_file, format='BMP')
    os.remove(adversarial_v_file)
    os.remove(original_file)
    

    if candi is not None:
        save_and_resize_image(candi, init_file)
        
        
        signdiff = torch.sign(adversarial_image)-torch.sign(candi)
        signdiff[signdiff!=0]=1
        save_and_resize_image(signdiff, initdiff_file)
        combined_image = concatenate_images([adversarial_file, init_file, initdiff_file])
        combined_image.save(initdiff_file, format='BMP')

        
    os.remove(init_file)
    os.remove(adversarial_file)
    return combined_file


def open_image(image_path):
    if os.name == 'nt':  # Windows
        os.startfile(image_path)
    elif os.name == 'posix':  # macOS and Linux
        subprocess.run(['open', image_path], check=True)

class ADBEvaluate():
    def __init__(self,PARA_TYPE=0):
        

        self.PARA_TYPE=PARA_TYPE #
        print(f"PARA_TYPE:{self.PARA_TYPE}")
        self.splitv = 0.4
        print(f"splitv={self.splitv}")
        ##################################################################################
        # # a,b,c,d
        # estimeated by ADBA
        if self.PARA_TYPE==0 or self.PARA_TYPE==2 :
            self.a_hat = 0.03133292769944518
            self.b_hat = 3.0659694403842903
            self.c_hat = 0.16755646970211466
            self.d_hat = 0.13403850261898806
        
        elif PARA_TYPE == 15:#
            tmp = [0.13227364, 1.35695908, 0.00135842, 0.02006667] 
        
        elif PARA_TYPE == 17:
            tmp = [0.02013212,  2.16136704, -0.04950291,  0.02321689] 
        
        elif PARA_TYPE == 22:
            tmp = [ 0.01401425,  2.37247315, -0.01584504,  0.02919938]
       
        if self.PARA_TYPE>=3:
            self.a_hat,self.b_hat,self.c_hat,self.d_hat= tmp[0],tmp[1],tmp[2],tmp[3]
        print(f"a_hat:{self.a_hat},b_hat:{self.b_hat},c_hat:{self.c_hat},d_hat:{self.d_hat}")




    # 
    def func_rho(self,r, a, b, c, d):
        return a / ((r + d) ** b) + c


   

    def find_midK_of_k1k2(self,k1, k2,method,paras=0): # 
        Sk1k2, error = quad(self.func_rho, k1, k2, args=(self.a_hat, self.b_hat, self.c_hat, self.d_hat))
        low, high, mid = k1, k2, 0
        Sk1mid = None
        while high - low > 1.0 / 600:
            mid = (low + high) / 2
            Sk1mid, error = quad(self.func_rho, k1, mid, args=(self.a_hat, self.b_hat, self.c_hat, self.d_hat))
            if Sk1mid < Sk1k2 / 2:
                low = mid
            else:
                high = mid
        return mid


    def next_ADB(self,r1, r2, aim_r, max_r, mod,method,paras=0): 
        if mod == 0:#ADBA
            return (r1 + r2) / 2.0
        if mod == 1:#ADBA-md
            k1, k2 = r1 / max_r, r2 / max_r
            kmid = self.find_midK_of_k1k2(1-k2, 1-k1,method=method,paras=paras)
            median = (1-kmid) * max_r
            return median