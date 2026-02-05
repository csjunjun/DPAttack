# DPAttack

## Overview

This repository provides an implementation of the paper *"Low-Cost Hard-Label Adversarial Attack From First Principles"*. It includes code and data for reproducing all experimental results presented in the paper.

## Setup Instructions

1. Install required Python packages based on the import statements in each file: pip install -r requirement.txt
3. Download the necessary datasets and model weights as described in the relevant sections below

## Reproducing Experimental Results

### 1. Fig. 3 - BFS Analysis

Generate and save perturbed images:
```bash
python baselines/attack_imagenet_others.py --attack_method BFS
```

Draw BFS curves:
```python
# In tools/OursAnalysis.py, run:
analyzeDCTSensitivity()
```

### 2. Fig. 4 - Clean Image Frequency Statistics

```python
# In tools/OursAnalysis.py, run:
analyzeCleanBDCT()
```

### 3. Table 1 - Pearson Correlation Analysis

First, conduct Experiments 1 and 2 to obtain preliminary results, then run:
```python
# In tools/OursAnalysis.py, run:
pearson()
```

### 4. Fig. 5 - Gradient Analysis

**Fig. 5(a)**: Cosine similarity between true gradient sign and block-wise averaged approximations

1. Obtain true gradient:
   ```bash
   python baselines/attack_imagenet_others.py --attack_method fgsmgrad
   ```

2. Generate gradient spatial prior analysis:
   ```python
   # In tools/OursAnalysis.py, run:
   gradSpatialPrior()
   ```

3. Draw figures:
   ```python
   # In tools/OursAnalysis.py, run:
   visgradSpatialPrior_bar()
   ```

**Fig. 5(b)**: Cosine similarity between true gradient sign and various initialization directions

```python
# In tools/OursAnalysis.py, run:
compareInitGradCossim()
```

### 5. Table 2 - Estimate Curvature κ

1. Clone PyHessian repository:
   ```bash
   git clone https://github.com/amirgholami/PyHessian.git
   ```
   Follow their installation instructions.

2. Place `tools/Hessian.py` in the downloaded PyHessian folder

3. Generate and save intermediate adversarial examples using query-based attack methods

4. Run the curvature estimation:
   ```python
   # In tools/Hessian.py, run to estimate the curvature of intermediate adversarial examples
   ```

### 6. Fig. 6 - Evolution of Cosine Similarity

Generate and save cosine similarity values:

- For "Ours":
  ```bash
  python tools/OursAnalysis.py --ablation 0 --binaryAnalyze 4
  ```

- For "Ours w/o PDO":
  ```python
  # In tools/OursAnalysis.py, run:
  main_ADBA() --ablation 1 --binaryAnalyze 4
  ```

- For "HRayS":
  ```bash
  python baselines/attack_imagenet_others.py --attack_method rays --binaryAnalyze 1
  ```

- For "ADBA":
  ```bash
  python baselines/ADBA.py --binaryAnalyze 7
  ```

Generate the evolution plot:
```python
# In tools/OursAnalysis.py, run:
cossimEvo()
```

### 7. Fig. 7 - Empirical Validation of Theorem 5

```python
# In tools/OursAnalysis.py, run:
callTheorem5()
```

### 8. Attack Results on Standard and Robust Models

Attack ImageNet, ImageNet-C, and ObjectNet with standard models, robust models, and CLIP:

- **Ours (Optimal Block Size)**: `Ours.py` (set blocksize to optimal value for `Ours_opt` results)
- **Ours with Dynamic Block Size Selection**: `OursDBS.py` (generates `Ours_dyn` results)
- **Ours with Dynamic Block Size Selection for CIFAR-10**: `OursCifar10DBS.py` (tools/utils.py getCifar10Testdata() to make cifar10 test data)

For other baseline methods, run the corresponding files in the `baselines/` directory:
```bash
git clone https://github.com/machanic/TangentAttack.git baselines/
```

#### ObjectNet

Download the dataset from https://objectnet.dev/download.html and place it in `data/objectnet/`. If the full dataset is too large, download only the file list from `data/objectnet/cliptest1000_withoutimagenetclasswithidx.txt`.

```bash
python OursImageNetDBS.py --datasource objectnet --victimmodel clip
```

#### PathMNIST

Install required package:
```bash
pip install medmnist
```

The provided checkpoint is trained following the repository: https://github.com/MedMNIST/MedMNIST

### 9. API Attacks


Install TencentCloud SDK:
```bash
pip install tencentcloud-sdk-python-intl-en
```
(See: https://github.com/TencentCloud/tencentcloud-sdk-python-intl-en)

Run the attack scripts in the `attackAPIs/` directory:

- Request API keys and secret keys from your API platform
- Set these variables in the code with your own credentials
- **Note**: When multiple threads call `getTempFilename()` simultaneously, temporary files may be overwritten. Use thread-specific names for temporary files to solve this issue.


### 10. Blacklight Defense

To attack models defended by Blacklight, set the `--defense` flag to 1 in the following files:

```bash
python OursBlacklight.py --defense 1
python baselines/ADBA.py --defense 1
python baselines/attack_imagenet_others.py --defense 1
```

### 11. Dense Prediction Tasks

#### Attack SAM on SA-1B Dataset

Navigate to the attack SAM directory:
```bash
cd attackSAM
python main.py
```

To test both our method and ADBA, clone the DarkSAM repository:
```bash
git clone https://github.com/CGCL-codes/DarkSAM.git
```
Follow their instructions to download the SA-1B dataset and SAM model.

#### Attack Object Detection on COCO


Download COCO validation set:
```bash
# Download COCO/val2017 from https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
```

The main entry point is `objectDetectionAttack.py`, which tests both our method and ADBA.


### 12. Ablation Studies

#### In Ours.py

- `--ablation`: Different search methods
  - `0`: PDO search
  - `1`: Dyadic fixed binary search (ADBA search)

- `--onlyone`: Different initialization methods
  - `0`: DDM
  - `1`: Our $d_n$
  - `2`: Our $\phi(d_r)$

#### In OursAblationInit.py

- `--ablation` with PDO search:
  - `2`: Uniform initialization
  - `3`: Gaussian initialization
  - `4`: Bar initialization
  - `5`: Random color initialization
  - `6`: Another image initialization

- `--ablation` with dyadic fixed binary search (ADBA search):
  - `21`: Uniform initialization
  - `31`: Gaussian initialization
  - `41`: Bar initialization
  - `51`: Random color initialization
  - `61`: Another image initialization

## Acknowledgments

This code is based on and modified from the ADBA paper implementation: https://github.com/BUPTAIOC/ADBA

We thank the authors for their foundational work.
