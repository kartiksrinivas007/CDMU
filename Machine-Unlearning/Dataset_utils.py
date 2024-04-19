from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import os
from utils import *
import torchvision
from tqdm import tqdm
# from utility_classes import ForgetObject



def stats_dataset(dataset):
    """
    Calculates the statistics of the images in the dataset (such as the mean and the standard deviation)
    
    Input: dataloader containing tensors of the form (N, C, H, W)
    Output: Torch tensor of mean and standard deviation each of size (C,)
    """    
    
    dataloader = DataLoader(dataset, batch_size = 64)
    
    img,_ = dataset[0]
    channels = img.size(0)
    sum_1 = sum_2 = batch_pixel_mean = squared_batch_pixel_mean = torch.zeros(channels)
    
    for batch_index, pair in enumerate(dataloader):
        batch, _ = pair
        batch_pixel_mean = torch.mean(batch, dim = (0, 2, 3))
        sum_1 = sum_1 + batch_pixel_mean
        squared_batch_pixel_mean  = torch.mean(torch.pow(batch, 2), dim = (0, 2, 3))
        sum_2 = sum_2 + squared_batch_pixel_mean
 
    
    mean = sum_1/batch_index
    std =  (sum_2/batch_index) - (torch.pow(mean, 2))
    
    return mean, torch.sqrt(std)
    

def return_domain_information(args, root_dir='./dataset/OHDS', batch_size = 32, train_val_split = 0.8):
    """
    Returns the information dictionary, which contains the train loaders val loaders and test loaders of each of the domains
    Also returns the dataset_names as a second output
    The forget object may or MAYNOT be created in this stage, depending on the dataset
    """
    information = {'dataset': [],  'trainset': [], 'testset': [], 'test_loader': [], 'train_loader': [], 'val_loader': [], 'means': [] , 'stds' : [], 'dataset_names': [], 'num_classes': -1}
    try:
        forget_labels = [int(label) for label in args.forget.split(',')]
    except:
        forget_labels = [1, 2]
        print("Wrong format for forget labels, defaulting to 1, 2")
    
    if (args.dset == 'office'):
        
        """
        Please note that this version is not upto date and certain inforation variables have not been made yet (for office Home)
        """
        information['dataset_names'] = ["Art", "Clipart", "Product", "Real_World"]
        information['means'].append(torch.tensor([0.6061, 0.5877, 0.5766]))
        information['means'].append(torch.tensor([0.5296, 0.5045, 0.4724]))
        information['means'].append(torch.tensor([0.5597, 0.5209, 0.4850]))
        information['means'].append(torch.tensor([0.5467, 0.5002, 0.4537]))
        information['stds'].append(torch.sqrt(torch.tensor([0.1264, 0.1239, 0.1264])))
        information['stds'].append(torch.sqrt(torch.tensor([0.1755, 0.1675, 0.1728])))
        information['stds'].append(torch.sqrt(torch.tensor([0.0978, 0.0948, 0.1011])))
        information['stds'].append(torch.sqrt(torch.tensor([0.0926, 0.0905, 0.0932])))
        
        fig = plt.figure(figsize= (10, 10))
        cols, rows = 2, 2

        domain_index = 1
        
        # apply the transform
        
        for domain_dir in os.listdir(root_dir):
            domain_directory = os.path.join(root_dir, domain_dir)
            if (os.path.isdir(domain_directory)):
                
                prelim_transform  = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
                                                        transforms.Normalize(information['means'][domain_index - 1], 
                                                                            information['stds'][domain_index - 1])])

                domain_dataset = ImageFolder(os.path.join(root_dir, domain_dir), transform=prelim_transform)
                
                
                if args.verbose:
                    print("Creating Dataset at root = ", domain_dataset.root)
                    print("Length of the Dataset = ", len(domain_dataset))
                
                
                # draw the images
                
                rand_img_index = torch.randint(len(domain_dataset), size = (1,))
                img, label = domain_dataset[rand_img_index]
                # print("Img is = ", img)
                # fig.add_subplot(rows, cols, domain_index)
                domain_index = domain_index + 1
                # plt.imshow(img.permute(1, 2, 0) , cmap= "gray") # gives the imshow error actually, because it is now normalized :(
                
            
                information['dataset'].append(domain_dataset)
                domain_loader = DataLoader(domain_dataset, batch_size=batch_size)
                
                # calc image stats 
                
                # mean, std_dev  = stats_dataset(domain_dataset)
                information['data_loader'].append(domain_loader)
                # print(mean, std_dev)

                # split them into 2

                domain_train_dataset, domain_test_dataset = random_split(domain_dataset, lengths=[train_val_split, (1 - train_val_split)])
                information['train_loader'].append(DataLoader(domain_train_dataset, batch_size= batch_size))
                information['test_loader'].append(DataLoader(domain_test_dataset, batch_size= batch_size))
                
                
                # breakpoint()
            pass
        
    elif (args.dset == 'digits'):
        information['dataset_names'] = ["mnist", "svhn", "usps", "syn", "mnistm"]
        information['num_classes'] = 10
        
        # now we have to get the dataset inside using the other files
        MEANS = [[0.1307, 0.1307, 0.1307], [0.4379, 0.4440, 0.4731], [0.2473, 0.2473, 0.2473], [0.4828, 0.4603, 0.4320], [0.4595, 0.4629, 0.4097]]
        STDS = [[0.3015, 0.3015, 0.3015], [0.1161, 0.1192, 0.1017], [0.2665, 0.2665, 0.2665], [0.1960, 0.1938, 0.1977], [0.1727, 0.1603, 0.1785]]

        im_size = (32, 32)
        
        # Prepare data
        transform_mnist = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[0], STDS[0])
            ])
        unnormalized_transform_mnist = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])

        transform_svhn = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[1], STDS[1])
            ])
        unnormalized_transform_svhn = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor()
            ])

        transform_usps = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[2], STDS[2])
            ])
        unnormalized_transform_usps = transforms.Compose([
                transforms.Resize(im_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])

        transform_synth = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[3], STDS[3])
            ])
        unnormalized_transform_synth = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor()
            ])

        transform_mnistm = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize(MEANS[4], STDS[4])
            ])
        unnormalized_transform_mnistm = transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor()
            ])

        """
        MNIST Dataset loading
        """
        #region 
        unnormalized_mnist_dset = torchvision.datasets.MNIST(root="./dataset/digits", train=True, transform=unnormalized_transform_mnist, download=True)
        information['dataset'].append(unnormalized_mnist_dset)
        mnist_trainset = torchvision.datasets.MNIST(root="./dataset/digits", train=True, transform=transform_mnist, download=True)
        
        mnist_trainset, mnist_valset = random_split(mnist_trainset, [args.train_val_split, 1 - args.train_val_split] )
        information['trainset'].append(mnist_trainset)
        mnist_train_loader = DataLoader(mnist_trainset, batch_size= args.batch, shuffle=True)
        information['train_loader'].append(mnist_train_loader)
        
        mnist_val_loader = DataLoader(mnist_valset, batch_size= args.batch, shuffle=True)
        information['val_loader'].append(mnist_val_loader)
        
        mnist_testset = torchvision.datasets.MNIST(root="./dataset/digits", train=False, transform=transform_mnist, download=True)
        mnist_test_loader = DataLoader(mnist_testset, batch_size= args.batch, shuffle=True)
        information['test_loader'].append(mnist_test_loader)
        information['testset'].append(mnist_testset)
        

        # endregion


        """
        SVHN dataset Loading
        """
        #region
        unnormalized_svhn_dataset = torchvision.datasets.SVHN(root="./dataset/digits", split='train', transform=unnormalized_transform_svhn, download=True)
        svhn_trainset = torchvision.datasets.SVHN(root="./dataset/digits", split='train', transform=transform_svhn, download=True)
        information['dataset'].append(unnormalized_svhn_dataset)
        
        svhn_trainset, svhn_valset = random_split(svhn_trainset, [args.train_val_split, 1 - args.train_val_split] )
        information['trainset'].append(svhn_trainset)
        svhn_train_loader = DataLoader(svhn_trainset, batch_size= args.batch, shuffle=True)
        information['train_loader'].append(svhn_train_loader)
        information['val_loader'].append(DataLoader(svhn_valset, batch_size= args.batch, shuffle=True))
        
        svhn_testset = torchvision.datasets.SVHN(root="./dataset/digits", split='test', transform=transform_svhn, download=True)
        information['test_loader'].append(DataLoader(svhn_testset, batch_size= args.batch, shuffle=True))
        information['testset'].append(svhn_testset)
        
        #endregion
        
        
        """
        USPS dataset loading
        """
        #region
        unnormalized_usps_trainset = torchvision.datasets.USPS(root="./dataset/digits", train=True, transform=unnormalized_transform_usps, download=True)
        information['dataset'].append(unnormalized_usps_trainset)
        
        usps_trainset = torchvision.datasets.USPS(root="./dataset/digits", train=True, transform=transform_usps, download=True)
        usps_trainset, usps_valset = random_split(usps_trainset, [args.train_val_split, 1 - args.train_val_split] )
        information['trainset'].append(usps_trainset)
        
        usps_train_loader = DataLoader(usps_trainset, batch_size= args.batch, shuffle=True)
        information['train_loader'].append(usps_train_loader)
        information['val_loader'].append(DataLoader(usps_valset, batch_size= args.batch, shuffle=True))
        
        usps_testset = torchvision.datasets.USPS(root="./dataset/digits", train=False, transform=transform_usps, download=True)
        usps_test_loader = DataLoader(usps_testset, batch_size= args.batch, shuffle=True)
        information['test_loader'].append(usps_test_loader)
        information['testset'].append(usps_testset)
        
        #endregion
        
        """
        Synthetic Digits Dataset Loading
        """
        #region
        unnormalized_synth_trainset     = ImageFolder('./dataset/digits/synthetic_digits/imgs_train', transform=unnormalized_transform_synth)
        information['dataset'].append(unnormalized_synth_trainset)
        
        synth_trainset     = ImageFolder('./dataset/digits/synthetic_digits/imgs_train', transform=transform_synth)
        synth_trainset, synth_valset = random_split(synth_trainset, [args.train_val_split, 1 - args.train_val_split] )
        information['trainset'].append(synth_trainset)
        
        synth_train_loader = DataLoader(synth_trainset, batch_size= args.batch, shuffle=True)
        information['train_loader'].append(synth_train_loader)
        information['val_loader'].append(DataLoader(synth_valset, batch_size= args.batch, shuffle=True))
        
        synth_testset     = ImageFolder('./dataset/digits/synthetic_digits/imgs_valid', transform=transform_synth)
        information['test_loader'].append(DataLoader(synth_testset, batch_size= args.batch, shuffle=True))
        information['testset'].append(synth_testset)
        
        #endregion
        
   
        """
        MNISTM Dataset Loading
        """
        #region

        unnormalized_mnistm_trainset     = ImageFolder('./dataset/digits/mnistm/train', transform=unnormalized_transform_mnistm)
        information['dataset'].append(unnormalized_mnistm_trainset)
    
        mnistm_trainset  = ImageFolder('./dataset/digits/mnistm/train', transform=transform_mnistm)
        mnistm_trainset, mnistm_valset = random_split(mnistm_trainset, [args.train_val_split, 1 - args.train_val_split] )
        information['trainset'].append(mnistm_trainset)
        mnistm_train_loader = DataLoader(mnistm_trainset, batch_size= args.batch, shuffle=True)
        information['train_loader'].append(mnistm_train_loader)
        mnistm_val_loader = DataLoader(mnistm_valset, batch_size= args.batch, shuffle=True)
        information['val_loader'].append(mnistm_val_loader)
        
        mnistm_testset    = ImageFolder('./dataset/digits/mnistm/test', transform=transform_mnistm)
        information['test_loader'].append(DataLoader(mnistm_testset, batch_size= args.batch, shuffle=True))
        information['testset'].append(mnistm_testset)
        
        #endregion

        # here you should create the forgetobjects of them both and then put them into the information maybe
        pass
    return information

if __name__ == "__main__":
    information = return_domain_information(root_dir='./dataset/OHDS/', batch_size=12)
    pass
