# F-Conv
Code of "Fourier Series Expansion Based Filter Parametrization for Equivariant Convolutions"  
[Paper link](https://ieeexplore.ieee.org/abstract/document/9851557)  
[Arxiv link](https://arxiv.org/submit/4780066/view)  
[Supplementary Material](https://github.com/XieQi2015/F-Conv/blob/main/Supplementary%20Material.pdf)

![Illustration of F-Conv](https://raw.githubusercontent.com/XieQi2015/ImageFolder/master/F-Conv/Fig4.jpg)

    MinistExp\    : Code for the experiments on minist dataset
    SRExp\        : Code for the experiments of image super resolution 
    F_Conv        : Code of the F-Conv proposed in our TPAMI paper
    B_Conv        : Code of the B-Conv, new version of F-Conv
    Supplementary Material.pdf: Supplementary Material
    
F-Convs are rotation equivariant convolutions with high representation accuracy, which can perform better on low-level computer vision tasks as comparsion with pervious rotation equivariant convolution methods.

Rotation symmetry on local features is an important structure characteristics of image, which can be hardly captured by commonly used CNN, as shown in the folliwing figure:

<img src="https://raw.githubusercontent.com/XieQi2015/ImageFolder/master/F-Conv/EqExample_1_new2.jpg" width="620">
(a) is a typical input cartoon image. (b) and (c) are outputs of randomly
initialized CNN and F-Conv, respectively, where the demarcated areas
are zoomed in 5 times for easy observation.

From the figure, we can also observe that the proposed F-Conv is expected to better maintain the symmetry of local features underlying the image, which should be help for the computer vision tasks.

Besides the output of F-Conv can be more stable than output of CNN when the input is rotated, as shown in the following two figures.

**CNN results:**

<img src="https://github.com/XieQi2015/ImageFolder/blob/master/F-Conv/CNN_tiny2.gif">

**F-Conv results:**

<img src="https://github.com/XieQi2015/ImageFolder/blob/master/F-Conv/FCNN_tiny2.gif">

**Usage:**    
For your CNN network, replace all the convolution layers with the proposed F-Conv layers. 

Specifically, for the first layer of network, use Fconv_PCA with ifIni=1. For example:

    #first layer of CNN

    import torch.nn as nn
    Conv_1 = nn.Conv_2d(c_in, c_out, kernel_size)

    #first layer of F-Conv

    import F_Conv as fn
    tranNum = 4 #2*pi/tranNum degree rotation equviariant 
    Conv_1 = fn.Fconv_PCA(kernel_size, c_in, c_out//tranNum, tranNum, ifIni=1) 
    # ifIni=1 is important

For the intermediate layer of network, use Fconv_PCA with ifIni=0. For example:

    #intermediate layer of CNN

    Conv_2 = nn.Conv_2d(c_in, c_out, kernel_size)

    #intermediate layer of F-Conv

    Conv_2 = fn.Fconv_PCA(kernel_size, c_in//tranNum, c_out//tranNum, tranNum, ifIni=0) 
    # ifIni=0 is important

For the output layer of network, use Fconv_PCA_out. For example:

    #output layer of CNN

    Conv_3 = nn.Conv_2d(c_in, c_out, kernel_size)

    #output layer of F-Conv

    Conv_3 = fn.Fconv_PCA_out(kernel_size, c_in//tranNum, c_out, tranNum)

More detail usage can be found in the subfolders

Citation:

    Qi Xie, Qian Zhao, Zongben Xu and Deyu Meng*. 
    Fourier Series Expansion Based Filter Parametrization for Equivariant Convolutions[J]. 
    IEEE transactions on pattern analysis and machine intelligence, 2022.
    
BibTeX:
    
    @article{xie2022FConv,
    title={Fourier Series Expansion Based Filter Parametrization for Equivariant Convolutions},
    author={Xie, Qi and Zhao, Qian and Xu, Zongben and Meng, Deyu},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2022},
    publisher={IEEE}
    }


