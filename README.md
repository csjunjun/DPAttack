# DPAttack

# setup

# objectnet
download dataset from https://objectnet.dev/download.html and put it to data/objectnet/, if it is too large, you can only download the filelists in data/objectnet/cliptest1000_withoutimagenetclasswithidx.txt

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
you have to git clone https://github.com/machanic/TangentAttack.git to the folder of "baselines"

# for real-world APIs
you need to request your own api key and secret key from the API platform. and then set these variables in the code as your own.
Note that if there are multiple threads simultaneously running, and calling the getTempFilename function, the save temporary files may be overlapped and leads to wrong result.
Naming these temporary files with different names for each thread can solve this problem.
install tencentcloud following the instructions from https://github.com/TencentCloud/tencentcloud-sdk-python-intl-en.git
