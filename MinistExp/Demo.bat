CALL conda activate pytorch
d:
cd D:\Documents\GitHub\F-Conv\MinistExp

%test the proposed method% 
python Rotated_MNIST_Main.py --dir NormalNet

%best result among 30 repetitions%
python Rotated_MNIST_Main.py --dir NormalNet_best_of_30_repetitions

%test the simple architectures in our paper% 
python Rotated_MNIST_simpleCase_Main.py --dir SimpleNet 

%retrain the proposed method% 
:python Rotated_MNIST_Main.py --dir retrain --mode train

%retrain the simple architectures in our paper% 
:python Rotated_MNIST_simpleCase_Main.py --dir SimpleNet_Retrain --mode train

PAUSE