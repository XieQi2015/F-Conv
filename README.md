# F-Conv
Code of Fourier Series Expansion Based Filter Parametrization for Equivariant Convolutions

![Illustration of F-Conv](https://raw.githubusercontent.com/XieQi2015/ImageFolder/master/F-Conv/Fig4.jpg)

    MinistExp\    : Code for the experiments on minist dataset
    SRExp\    : Code for the experiments of image super resolution 
    
F-Convs are rotation equivariant convolutions with high representation accuracy, which can perform better on low-level computer vision tasks as comparsion with pervious rotation equivariant convolution methods.

Rotation symmetry on local features is a important structure characteristics of image, which can be hardly captured by commonly used CNN, as shown in the folliwing Fig. 1:

![Illustration of F-Conv 2](https://raw.githubusercontent.com/XieQi2015/ImageFolder/master/F-Conv/EqExample_1_new2.jpg)
Fig. 1. (a) A typical input cartoon image. (b)-(c) Outputs of randomly
initialized CNN and F-Conv, respectively, where the demarcated areas
are zoomed in 5 times for easy observation.

From Fig. 1, we can also observe that the proposed F-Conv is expected to better maintain the symmetry of local features underlying the image, which should be help for the computer vision tasks.

Besides the output of F-Conv can be more stable than output of CNN, as  shown in the following two figures:
![Illustration of F-Conv 3](https://github.com/XieQi2015/ImageFolder/blob/master/F-Conv/CNN_small2.gif)
![Illustration of F-Conv 4](https://github.com/XieQi2015/ImageFolder/blob/master/F-Conv/FCNN_small2.gif)


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

