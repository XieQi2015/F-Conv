# Usage of The Codes
 
Folder structure:

    Data\    : data folder, the Div2k data should be download here 
    src\     : Main codes
    Demo.txt : Comands for calling all the experiments
 
Usage:

The code is built on [EDSR (PyTorch)](https://github.com/sanghyun-son/EDSR-PyTorch) and has been tested on both Ubuntu and Windows.
please refer [EDSR (PyTorch)](https://github.com/sanghyun-son/EDSR-PyTorch) for more usage details.

We used DIV2K dataset to train our model. Please download it from [here (7.1GB)](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar). Unpack the tar file to Data folder before using the codes, or Unpack the tar file to any place you want. Then, change the dir_data argument in src/option.py to the place where DIV2K images are located.

We used Urban100, B100, Set14 and Set5 dataset for testing, which can be download from [benchmark datasets (250MB)](https://cv.snu.ac.kr/research/EDSR/benchmark.tar), and Unpack the tar file to the Data folder.


To train and test the proposed method, Cd to 'src', and following scripts are example for training and testing, respectively:
    
    python main.py --model RDN_fcnn --scale 2 --save RDN_fcnn_x2  --res_scale 0.1 --ini_scale  0.1 --batch_size 16  --patch_size 64 --G0 8 --kernel_size 5 --epochs 150 --decay 3-100-130 --lr 4e-4
    python main.py --text_only --model RDN_fcnn --scale 2 --save RDN_fcnn_x2  --res_scale 0.1 --ini_scale  0.1 --batch_size 16  --patch_size 64 --G0 8 --kernel_size 5 --epochs 150 --decay 3-100-130 --lr 4e-4

More examples can be found in Demo.txt.
 
 Citation:

    Qi Xie, Qian Zhao, Zongben Xu and Deyu Meng*. 
    Fourier Series Expansion Based Filter Parametrization for Equivariant Convolutions[J]. 
    IEEE transactions on pattern analysis and machine intelligence, 2022.
    
    BibTeX:
    
    @article{xie2020MHFnet,
    title={Fourier Series Expansion Based Filter Parametrization for Equivariant Convolutions},
    author={Xie, Qi and Zhao, Qian and Xu, Zongben and Meng, Deyu},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2020},
    publisher={IEEE}
    }
