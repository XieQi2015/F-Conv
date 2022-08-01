
import torch 
import os
import argparse
import scipy.io as sio    
from SteerableCNN_XQ import MinstSteerableCNN_simple
from DataLoader import MnistRotDataset 
from torchvision.transforms import RandomRotation
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
#from MyLib import rotate_im
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dir',type = str, default = 'SimpleNet' )
# parser.add_argument('--testModel', type = str, default='./Model/Model_best_red.pt')
parser.add_argument('--device', type = str, default='0')
parser.add_argument('--mode',type = str, default = 'test' )
parser.add_argument('--weight_decay', type = float, default = 1e-2)
parser.add_argument('--InP', type = int, default = 4)
args = parser.parse_args()

mode = args.mode
tranNum  = 8
iniEpoch = 0
maxEpoch = 100
modelDir = os.path.join('./Models/'+ args.dir+ '/')
resultDir = os.path.join('./Results/'+ args.dir+ '/')
print(modelDir)
testModel = os.path.join(modelDir,'Model_best.pt')
testEvery = 1
saveEveryStep = False
use_test_time_augmentation = False
use_train_time_augmentation = False


ifshow = 0 # if show the Feature maps



device = 'cuda:'+args.device if torch.cuda.is_available() else 'cpu'

model = MinstSteerableCNN_simple(10,tranNum).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=args.weight_decay) 
# milestone = [15,30,60,90,300]
milestone = [30,60,150,300]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=0.2)
# milestone = [30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60]
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=0.8)

def test_with_aug(test_loader,model,use_test_time_augmentation):
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for i, (x, t) in enumerate(test_loader):
            out = None
            if use_test_time_augmentation:
                #Run same sample with different orientations through network and average output
                rotations = [-15,0,15]
            else:
                rotations = [0]
                    
            for rotation in rotations:
                for i in range(x.size(0)):
                    im = x[i,:,:,:].data.cpu().numpy().squeeze()
                    # im = rotate_im(im, rotation)
                    im = im.reshape([1,28,28])
                    x[i,:,:,:] = torch.FloatTensor(im)
                x = x.to(device)
                t = t.to(device)
                if out is None:
                    out = torch.nn.functional.softmax(model(x), dim = 1)
                else:
                    out+= torch.nn.functional.softmax(model(x), dim = 1)
            out/= len(rotations)
            _, prediction = torch.max(out.data, 1)
            total += t.shape[0]
            correct += (prediction == t).sum().item()
        return correct/total*100  


totensor = ToTensor()    
if mode == 'train':
    
    try:
        os.makedirs(modelDir)
    except OSError:
        pass
    
    try:
        os.makedirs(resultDir)
    except OSError:
        pass

    if iniEpoch:
        # load the previous trained model
        model.train()
        checkpoint = torch.load(os.path.join(modelDir, 'model_'+str(iniEpoch) + '.pt'))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print('loaded checkpoints, epoch {:d}'.format(checkpoint['epoch']))
        
    if use_train_time_augmentation:
        train_transform = Compose([
            Resize(87),
            RandomRotation(360, resample=Image.BILINEAR, expand=False),
            Resize(28),
            totensor,
        ])
    else:
        train_transform = Compose([
            totensor,
        ])
    mnist_train = MnistRotDataset(mode='train', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True)
    
    
    test_transform = Compose([
        totensor,
    ])
    mnist_test = MnistRotDataset(mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100)
    loss_function = torch.nn.CrossEntropyLoss()
    
    
    best_acc = 0
    ifshow = 0
    for epoch in range(iniEpoch,maxEpoch):
        model.train()
        total = 0
        correct = 0 
        # one epoch training
        for i, (x, t) in enumerate(train_loader):          
            optimizer.zero_grad()    
            x = x.to(device)
            t = t.to(device)    
            y = model(x, ifshow)    
            loss_class = loss_function(y, t)
            l1_regularization = 0
            loss = loss_class + l1_regularization
            loss.backward()#retain_graph=True
            optimizer.step()
            _, prediction = torch.max(y.data, 1)
            total += t.shape[0]
            correct += (prediction == t).sum().item()
        train_acc = correct/total*100  
        print(f"epoch {epoch} | train accuracy: {train_acc}")  
        scheduler.step()
        if epoch == 200:
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-3) 
            milestone = [90]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=0.1)
        
        # test the model and save the best model     
        if epoch % testEvery == 0 and epoch>30:
            test_acc = test_with_aug(test_loader,model,use_test_time_augmentation)
            print(f"The test accuracy: {test_acc}")
            print('The test error is %.5f' %(100-test_acc))
            if test_acc > best_acc:
                best_acc = test_acc
                model.train()
                save_path_model = os.path.join(modelDir, 'Model_best.pt')
                torch.save(model.state_dict(), save_path_model)
                sio.savemat(resultDir+'acc.mat',{'acc':best_acc})
            print('The best error is %.5f' %(100-best_acc))
            print('=================================================')    
        
        # save the model     
        if saveEveryStep:    
            save_path_model = os.path.join(modelDir, 'model_'+str(epoch+1)+ '.pt')
            torch.save({
                'epoch': epoch+1,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'model_state_dict':model.state_dict()
                }, save_path_model)
else: 
    # model.load_state_dict(torch.load(testModel))
    model.load_state_dict(torch.load(testModel,map_location=device))
    test_transform = Compose([
        totensor,
    ])
    mnist_test = MnistRotDataset(mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100)
    
    test_acc = test_with_aug(test_loader,model,use_test_time_augmentation)
    print(f"epoch {iniEpoch} | test error: {100-test_acc}")
