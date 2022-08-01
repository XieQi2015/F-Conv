# Usage of The Codes
 
Folder structure:

    Data\    : data folder, the Div2k data should be download here 
    src\     : Main codes
    Demo.txt : Comands for calling all the experiments
 
Usage:
 
To train and text the proposed method, followings are examples:

    python main.py --model RDN_fcnn --scale 2 --save RDN_fcnn_x2  --res_scale 0.1 --ini_scale  0.1 --batch_size 16  --patch_size 64 --G0 8 --kernel_size 5 --epochs 150 --decay 3-100-130 --lr 4e-4
    python main.py --text_only --model RDN_fcnn --scale 2 --save RDN_fcnn_x2  --res_scale 0.1 --ini_scale  0.1 --batch_size 16  --patch_size 64 --G0 8 --kernel_size 5 --epochs 150 --decay 3-100-130 --lr 4e-4

More examples can be found in Demo.txt
 
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
