# Usage of The Codes
 
Folder structure:

    Data\    : data folder
        |-- mnist_rotation_new.zip : minist data, Unzip before use. 
    Models\  : Trained result
        |-- NormalNet\      : Trained parameters
        |-- NormalNet_best_of_30_repetitions\     : Best model parameter of 30 random repetitions
        |-- SimpleNet\      : Trained parameters of the simple network in the paper
    DataLoader.py       : The data reading and preparing code
    F_Conv.py           : Core codes of the equivariant convolutions of F-Conv
    MyLib.py            : Some used code
    Rotated_MNIST_Main.py             : Main code of experiment
    Rotated_MNIST_simpleCase_Main.py  : Main code of experiment of the simple case in the paper
    SteerableCNN_XQ.py                : F-Conv based Networks 
    Demo.bat                          : Demo for running all the experiments
 
Usage:

To run testing with the example data, you can just run Rotated_MNIST_Main.py which is equivariant under reflections call:

  python Rotated_MNIST_Main.py --dir NormalNet

To test the best model trained by us, one can call:

  python Rotated_MNIST_Main.py --dir NormalNet_best_of_30_repetitions
  
To train the  model, one can call:

  python Rotated_MNIST_Main.py --dir retrain --mode train
 
To test and train the simple cases, one call:

  python Rotated_MNIST_simpleCase_Main.py --dir SimpleNet 
  python Rotated_MNIST_simpleCase_Main.py --dir SimpleNet_Retrain --mode train
  
