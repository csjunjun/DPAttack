import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
from darksam_attack import choose_dataset
from atk_setting import *
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent

sys.path.append(str(root_path))

try:
    from pycocotools.mask import decode
except ImportError:
    print('>> [error] missing lib "pycocotools", run "pip install pycocotools" first!!')
    raise
DATA_ROOT = DATASET_PATH['sam']
HIST_FILE = OUT_PATH / 'atk_sam.json'
transform = transforms.Compose([transforms.ToTensor()])

def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "w", encoding='utf8', )
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass
    fileName = datetime.now().strftime('%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    return fileName

def collate_fn(batch):
    images, prompts_list, mask_gt_list, sample_id  = zip(*batch)
    return images, prompts_list, mask_gt_list, sample_id

def run(args,path_id, custom_dataset):
    sam = load_sam(args.M)
    sam = sam.to(device)
    sam_fwder = SamForwarder(sam)
    sam_fwder = sam_fwder.to(device)
    loss_fn = F.mse_loss
    s = time()
    iou_sum_adv, iou_cnt = 0.0, 0
    iou_sum_img = 0.0
    interrupted = False
    batch_size = 1
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
    denorm = lambda x: sam_fwder.denorm_image(x) / 255.0
    # if path_id == "":
    #     print("us ours")
    # else:
    #     uap_save_path = path_id
    #     uap = torch.load(uap_save_path, map_location=device)
    for batch in tqdm(data_loader):
        images, p_list, mask_gt_list, sample_ids = batch
        mask_gt = mask_gt_list[0]
        P = p_list[0]
        for image in images:
            X = sam_fwder.transform_image(image)
            X = X.to(device)
            benign_img = (denorm(X) * 255).byte().div(255)
            benign_img = benign_img.to(device)
            with torch.no_grad():
                logits_clean, _ = sam_fwder.forward(benign_img, *P)
            logits_clean = logits_clean.to(device)
            mask_clean = logits_clean > sam_fwder.model.mask_threshold
            mask_clean = mask_clean.to(device)
            iou_benign_img = get_iou_auto(mask_clean.cpu().detach().numpy(), mask_gt)
            iou_benign_img_percentage = iou_benign_img * 100
            print(f"iou_benign_img: {iou_benign_img_percentage:.2f} %")
            if iou_benign_img <args.clean_lim:
                continue
            if args.test_method=="clean":

                # adv_img = benign_img + uap
                # adv_img = torch.clamp(adv_img, 0, 1)
                if iou_benign_img >=args.clean_lim:
                    iou_cnt += 1
                iou_sum_img = iou_sum_img + iou_benign_img
                if iou_cnt == args.test_num:
                    exit()

                continue
            elif args.test_method=="ADBA":
                
                success, query, _, linfdis, l2dis,adversarial_image,iou_adv_img =\
                        ATK_ADBA(sam_fwder, benign_img, None, iou_cnt, args.eps,8, args.budget,get_iou_auto,mask_gt,P,iouThreshold=args.iouthresh)
                
            elif args.test_method=="Ours":
                success, query, _, linfdis, l2dis,adversarial_image,iou_adv_img =\
                        ATK_ADBA(sam_fwder, benign_img, None, iou_cnt, args.eps,8, args.budget,get_iou_auto,mask_gt,P,iouThreshold=args.iouthresh,args=args)
            elif args.test_method=="OursDy":
                success, query, _, linfdis, l2dis,_,_,_,_,_,_,_,iou_adv_img =\
                        ATK_ADBA("",sam_fwder, benign_img, iou_cnt, None,None,args.eps,8,np.inf,None,[],"",get_iou_auto,mask_gt,P,iouThreshold=args.iouthresh,args=args)
            
            if success == 1 and query <= args.budget:
                print(f"img {iou_cnt} attack success, query: {query}, linfdis: {linfdis}, l2dis: {l2dis},iou_adv_img:{iou_adv_img}")
                iou_cnt += 1
                iou_sum_img = iou_sum_img + iou_benign_img
                iou_sum_adv = iou_sum_adv + iou_adv_img
            else:
                iou_cnt += 1
                print(f"img {iou_cnt} attack fail, query: {query}, linfdis: {linfdis}, l2dis: {l2dis},iou_adv_img:{iou_adv_img}")
            #del logits_clean, benign_img, X, mask_clean, logits_hat, mask_hat, adv_img
            if iou_cnt == args.test_num:
                print("reach test num")
                exit()

       
    miou_img = 0.0 if iou_cnt == 0 else (iou_sum_img / iou_cnt)
    miou_adv = 0.0 if iou_cnt == 0 else (iou_sum_adv / iou_cnt)
    print(f'>> miou_benign_img: {miou_img}, >> miou_adv_img: {miou_adv}, >> iou_cnt:{iou_cnt}')
    return iou_cnt,miou_img, miou_adv
   

def get_parser() -> ArgumentParser:
    from atk_setting import get_parser as get_base_parser
    parser = get_base_parser()
    parser.add_argument('--limit_img', default=500, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--budget', default=50, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--iouthresh', default=0.5, type=float, help='limit run image count, set -1 for all')

    parser.add_argument('--train_dataset', default='SA1B')
    parser.add_argument('--test_dataset', default='SA1B')
    parser.add_argument('--sta', choices=['train', 'test'], default='test', help='station of comm')
    parser.add_argument('--train_prompts', choices=['bx', 'pt'], default='pt', help='type of prompts (box or point)')
    parser.add_argument('--test_prompts', choices=['bx', 'pt'], default='pt', help='type of prompts (box or point)')
    parser.add_argument('--test_method', choices=['clean', 'ADBA','Ours',"OursDy"], default='OursDy', help='type of test methods')

    parser.add_argument('--eps', default=0.05, type=float)
    parser.add_argument('--seed', default=100, type=int, help='rand seed')
    parser.add_argument('--save', default='True', type=bool, help='save the csv')
    parser.add_argument('--train_num', default=100, type=int)
    parser.add_argument('--test_num', default=500, type=float)
    parser.add_argument('--clean_lim', default=0.5, type=float, help='clean iou lim')
    parser.add_argument('--M', default='vit_b', choices=SAM_CKPTS.keys(), help='model checkpoint')


    #for Ours attack method
    parser.add_argument('--zerosign', default=1, type=int,
                        help='')
    parser.add_argument('--earlyexit', default=0, type=int,
                        help='Dataset')

    parser.add_argument('--replace', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--useadba', default=1, type=int,
                        help='Dataset')
    parser.add_argument('--binaryAnalyze', default=0, type=int,
                        help='Dataset')
    parser.add_argument('--onlyone', default=0, type=int,
                        help='0::both,1:only std sample;2:only low color square;.')
    parser.add_argument('--ablation', default=0, type=int,
                        help='0::None,1:adbasearch.')
    parser.add_argument('--lowtype', default="rcolor", type=str,
                        help='dct,bar,rcolor,std') 
    parser.add_argument('--dwtlevel', default=4, type=float,
                        help='0:no;1;2')  
    parser.add_argument('--dctTrunc', default=4, type=int,
                        help='0:no;1;2')  
    parser.add_argument('--lowfilter', default=0, type=int,
                        help='0:dwt;1:bdct')  


    parser.add_argument('--freqratio', default=4, type=int,
                        help='freqratio:1-64')  
    parser.add_argument('--color', default=0, type=int,
                        help='freqratio:1-64')  

    parser.add_argument('--blocksize', default=4, type=int,
                        help='0:no defense;1:blacklight')   
    parser.add_argument('--deviceid', default="7", type=str,
                        help='attack batch size.')
    
    parser.add_argument('--binaryM', default=1, type=int,
                        help='binary search mod, mid 0 or median 1.')
    parser.add_argument('--early_stop', default=1, type=int,
                        help='early_stop')
    parser.add_argument('--initDir', default=-1e5, type=int,
                        help='initial direction, 1,-1,and 0 for random;-1e5 for freqSearch init')
    parser.add_argument('--channels', default=3, type=int,
                        help='output channels, 3 for max channels, 1 for 1 channel for all datas')
    parser.add_argument('--offspringN', default=2, type=int,
                        help='offspring diretion num in new iteration')
    parser.add_argument('--targeted', default=0, type=int,
                        help='targeted or untargeted')
    parser.add_argument('--norm', default='np.linf', type=str,
                        help='Norm for attack, linf only')
    parser.add_argument('--batch', default=1, type=int,
                        help='attack batch size.')
    parser.add_argument('--early', default='1', type=str,
                        help='early stopping (stop attack once the adversarial example is found)')

    parser.add_argument('--defense', default=0, type=int,
                        help='0:no defense;1:blacklight')
    parser.add_argument('--stepp', default=1, type=float,
                        help='0:no defense;1:blacklight')    
    parser.add_argument('--mu', default=0, type=float,
                        help='0:no defense;1:blacklight')   
    parser.add_argument('--initystd', default=0, type=float,
                        help='0:no defense;1:blacklight')  
    parser.add_argument('--initcbstd', default=0, type=float,
                        help='0:no defense;1:blacklight')  
    parser.add_argument('--initcrstd', default=0, type=float,
                        help='0:no defense;1:blacklight')  

    parser.add_argument('--paratype', default=17, type=int,
                        help='0:no defense;1:blacklight')  

    return parser

def get_args(parser: ArgumentParser) -> Namespace:
    from atk_setting import get_args as get_base_args
    args = get_base_args(parser)
    args.f = None
    args.D = 'sam'
    args.fps = -1
    args.debug = False
    return args

if __name__ == '__main__':
    parser = get_parser()
    args = get_args(parser)
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.test_method=="ADBA":
        from models.ADBASAM import ATK_ADBA
    elif args.test_method=="Ours":
        from models.OursSAM import ATK_ADBA
    elif args.test_method=="OursDy":
        from models.OursDynaStartSAM import ATK_ADBA
    if args.test_dataset == 'CITY':
        args.clean_lim = 0.12
    log_save_path = os.path.join('result', 'test', 'log')
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
    now_time = make_print_to_file(path=log_save_path)
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
    with open(log_save_path + '/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    custom_dataset = choose_dataset(args.test_dataset, args)
    #custom_dataset = None
    #path_id = f"uap_file/{args.test_dataset}.pth"
    path_id = ""
    test_num, miouimg, miouadv= run(args, path_id, custom_dataset)
    print(f":: miouimg: {miouimg * 100:.2f} %, miouadv: {miouadv * 100:.2f} %")

    
