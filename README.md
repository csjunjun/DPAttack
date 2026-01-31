# DPAttack


# for PathMNIST
pip install medmnist

The provided checkpoint is trained by following the repository https://github.com/MedMNIST/MedMNIST


# for attack SAM on SA-1B
cd attackSAM main.py to test ours and ADBA
git clone https://github.com/CGCL-codes/DarkSAM.git
follow their instructions to download the SA1B dataset and SAM model.

# for attack Object Detection on COCO
entrance file is objectDetectionAttack.py, test ours and ADBA
you need to download COCO/val2017 from https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset

# for evaluating baselines
go to the baselines folder
the entrance for ADBA is named with ADBAxxx.py others entraince are in the attack_{dataset}_others.py
