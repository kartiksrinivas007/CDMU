import torch
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Subset
import random
import os
import numpy as np
from MUDatasetObject import MUDatasetObject
# from cifar10c_dataset import CIFAR10C, CIFAR10C_preprocessed


def make_forget_train_loader(args, index, datasets, forget_condition):
    """
    Function takes in the index, finds the dataset and returns the forget and remember loaders of each one respectively,
    for the train and the test both, so there should be four loaders per dataset in total, and the fifth, sixth for the total one if necessary
    
    Perhaps making an object for the dataset seems to be the optimal solution, each one having certain properties
    """
    train_ds, test_ds = datasets[index]
    fg_labels, rm_labels = forget_condition
    
    # breakpoint()
    
    try:
        train_id_forget = np.where([i in fg_labels for i in train_ds.targets])[0]
        train_id_remember = np.where([i in rm_labels for i in train_ds.targets])[0]
        test_id_forget = np.where([i in fg_labels for i in test_ds.targets])[0]
        test_id_remember = np.where([i in rm_labels for i in test_ds.targets])[0]
 
    except:
        # breakpoint()
        train_id_forget = np.where([i in fg_labels for i in train_ds.labels])[0]
        train_id_remember = np.where([i in rm_labels for i in train_ds.labels])[0]
        test_id_forget = np.where([i in fg_labels for i in test_ds.labels])[0]
        test_id_remember = np.where([i in rm_labels for i in test_ds.labels])[0]

        # ids = [train_id_forget, train_id_remember, test_id_forget, test_id_remember, train_all_id, test_all_id]
        
    train_all_id = np.arange(len(train_ds))
    test_all_id = np.arange(len(test_ds))
    # ids = [train_id_forget, train_id_remember, test_id_forget, test_id_remember, train_all_id, test_all_id]
    train_ids = [train_id_forget, train_id_remember, train_all_id]
    test_ids = [test_id_forget, test_id_remember, test_all_id]
    
    train_loaders = [torch.utils.data.DataLoader(Subset(train_ds, id), batch_size=args.batch, shuffle=True) for id in train_ids]
    test_loaders = [torch.utils.data.DataLoader(Subset(test_ds, id), batch_size = args.batch, shuffle=True) for id in test_ids]
    
    try:
        tsne_remember_loader = torch.utils.data.DataLoader(Subset(Subset(train_ds, train_id_remember), np.arange(2000)), batch_size=100, shuffle=True)
        tsne_forget_loader = torch.utils.data.DataLoader(Subset(Subset(train_ds, train_id_forget), np.arange(2000)), batch_size=100, shuffle=True)
    except:
        print("Not enough data to make a tsne loader for this dataset index = ", index)
    tsne_loader = [tsne_forget_loader, tsne_remember_loader]
    
    
    muds = MUDatasetObject(train_loaders, test_loaders, tsne_loader)  
    # make loaders using these id's for all of the possible datasets, and then return them in an array
    # loaders = [torch.utils.data.DataLoader(Subset(train_ds,))for id in ids]
    return muds
    pass


def make_unlearning_loaders(args, dataset_names, datasets, forget_condition):
    # need to find acondition on the basis of which the sbset can be split and the id's can be obtained
    source_index = dataset_names.index(args.source)
    target_index = dataset_names.index(args.target)
    
    print(f"Source Dataset is {args.source} and Target Dataset is {args.target} ")
    # fidn the two types of id's that can be found
    source_mud = make_forget_train_loader(args, source_index, datasets, forget_condition)
    target_mud = make_forget_train_loader(args, target_index, datasets, forget_condition)
    return source_mud, target_mud
    pass


def prepare_data(args, im_size):

    if args.dataset == 'digits':

        MEANS = [[0.1307, 0.1307, 0.1307], [0.4379, 0.4440, 0.4731], [0.2473, 0.2473, 0.2473], [0.4828, 0.4603, 0.4320], [0.4595, 0.4629, 0.4097]]
        STDS = [[0.3015, 0.3015, 0.3015], [0.1161, 0.1192, 0.1017], [0.2665, 0.2665, 0.2665], [0.1960, 0.1938, 0.1977], [0.1727, 0.1603, 0.1785]]

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

        dataset_names = ["mnist", "svhn", "usps", "syn", "mnistm"]
        datasets = []
                
        unnormalized_mnist_trainset = torchvision.datasets.MNIST(root="./digit_data", train=True, transform=unnormalized_transform_mnist, download=True)
        mnist_trainset = torchvision.datasets.MNIST(root="./digit_data", train=True, transform=transform_mnist, download=True)
        mnist_testset = torchvision.datasets.MNIST(root="./digit_data", train=False, transform=transform_mnist, download=True)
        datasets.append((mnist_trainset, mnist_testset))

        # print(f'MNIST: {len(mnist_testset)}')

    
        unnormalized_svhn_trainset = torchvision.datasets.SVHN(root="./digit_data", split='train', transform=unnormalized_transform_svhn, download=True)
        svhn_trainset = torchvision.datasets.SVHN(root="./digit_data", split='train', transform=transform_svhn, download=True)
        svhn_testset = torchvision.datasets.SVHN(root="./digit_data", split='test', transform=transform_svhn, download=True)
        datasets.append((svhn_trainset, svhn_testset))
        # print(f'SVHN: {len(svhn_testset)}')

        unnormalized_usps_trainset = torchvision.datasets.USPS(root="./digit_data", train=True, transform=unnormalized_transform_usps, download=True)
        usps_trainset = torchvision.datasets.USPS(root="./digit_data", train=True, transform=transform_usps, download=True)
        usps_testset = torchvision.datasets.USPS(root="./digit_data", train=False, transform=transform_usps, download=True)
        datasets.append((usps_trainset, usps_testset))
        # print(f'USPS: {len(usps_testset)}')

        unnormalized_synth_trainset     = ImageFolder('./digit_data/synthetic_digits/imgs_train', transform=unnormalized_transform_synth)
        synth_trainset     = ImageFolder('./digit_data/synthetic_digits/imgs_train', transform=transform_synth)
        synth_testset     = ImageFolder('./digit_data/synthetic_digits/imgs_valid', transform=transform_synth)
        datasets.append((synth_trainset, synth_testset))
        
        # print(f'SYNTH: {len(synth_testset)}')


        unnormalized_mnistm_trainset     = ImageFolder('./digit_data/mnistm/train', transform=unnormalized_transform_mnistm)
        mnistm_trainset     = ImageFolder('./digit_data/mnistm/train', transform=transform_mnistm)
        mnistm_testset    = ImageFolder('./digit_data/mnistm/test', transform=transform_mnistm)
        datasets.append((mnistm_trainset, mnistm_testset))

 
        # has been hardcoded for now, but should be changed later, to take it with the argument!,
        forgetting_labels = [0,1]
        remembering_labels = [2,3,4,5,6,7,8,9]
        
        
        # breakpoint()
        source_mud, target_mud = make_unlearning_loaders(args, dataset_names, datasets, (forgetting_labels,remembering_labels))

        # breakpoint()
        
        return source_mud, target_mud # note that this path returns directly!

    elif args.dataset == 'retina':

        MEANS = [[0.5594, 0.2722, 0.0819], [0.7238, 0.3767, 0.1002], [0.5886, 0.2652, 0.1481], [0.7085, 0.4822, 0.3445]]
        STDS = [[0.1378, 0.0958, 0.0343], [0.1001, 0.1057, 0.0503], [0.1147, 0.0937, 0.0461], [0.1663, 0.1541, 0.1066]]

        # data_base_path = './data/segmented_retina'
        data_base_path = './data/retina_balanced'
        
        transform_unnormalized = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor()
        ])
        
        # Drishti
        transform_drishti = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor(),
                transforms.Normalize(MEANS[0], STDS[0])
        ])
        drishti_train_path = os.path.join(data_base_path, 'Drishti', 'Training')
        drishti_test_path = os.path.join(data_base_path, 'Drishti', 'Testing')
        unnormalized_drishti_trainset = ImageFolder(drishti_train_path, transform=transform_unnormalized)
        drishti_trainset = ImageFolder(drishti_train_path, transform=transform_drishti)
        drishti_testset = ImageFolder(drishti_test_path, transform=transform_drishti)
        
        # kaggle
        transform_kaggle = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor(),
                transforms.Normalize(MEANS[1], STDS[1])
        ])
        kaggle_train_path = os.path.join(data_base_path, 'kaggle_arima', 'Training')
        kaggle_test_path = os.path.join(data_base_path, 'kaggle_arima', 'Testing')
        unnormalized_kaggle_trainset = ImageFolder(kaggle_train_path, transform=transform_unnormalized)
        kaggle_trainset = ImageFolder(kaggle_train_path, transform=transform_kaggle)
        kaggle_testset = ImageFolder(kaggle_test_path, transform=transform_kaggle)
        
        # RIM
        transform_rim = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor(),
                transforms.Normalize(MEANS[2], STDS[2])
        ])
        rim_train_path = os.path.join(data_base_path, 'RIM', 'Training')
        rim_test_path = os.path.join(data_base_path, 'RIM', 'Testing')
        unnormalized_rim_trainset = ImageFolder(rim_train_path, transform=transform_unnormalized)
        rim_trainset = ImageFolder(rim_train_path, transform=transform_rim)
        rim_testset = ImageFolder(rim_test_path, transform=transform_rim)
        
        # refuge
        transform_refuge = transforms.Compose([
                transforms.Resize(im_size),            
                transforms.ToTensor(),
                transforms.Normalize(MEANS[3], STDS[3])
        ])
        refuge_train_path = os.path.join(data_base_path, 'REFUGE', 'Training')
        refuge_test_path = os.path.join(data_base_path, 'REFUGE', 'Testing')
        unnormalized_refuge_trainset = ImageFolder(refuge_train_path, transform=transform_unnormalized)
        refuge_trainset = ImageFolder(refuge_train_path, transform=transform_refuge)
        refuge_testset = ImageFolder(refuge_test_path, transform=transform_refuge)
        

        #####################################
        Drishti_train_loader = torch.utils.data.DataLoader(drishti_trainset, batch_size=args.batch, shuffle=True)
        Drishti_test_loader = torch.utils.data.DataLoader(drishti_testset, batch_size=args.batch, shuffle=False)

        kaggle_train_loader = torch.utils.data.DataLoader(kaggle_trainset, batch_size=args.batch, shuffle=True)
        kaggle_test_loader = torch.utils.data.DataLoader(kaggle_testset, batch_size=args.batch, shuffle=False)

        rim_train_loader = torch.utils.data.DataLoader(rim_trainset, batch_size=args.batch, shuffle=True)
        rim_test_loader = torch.utils.data.DataLoader(rim_testset, batch_size=args.batch, shuffle=False)

        refuge_train_loader = torch.utils.data.DataLoader(refuge_trainset, batch_size=args.batch, shuffle=True)
        refuge_test_loader = torch.utils.data.DataLoader(refuge_testset, batch_size=args.batch, shuffle=False)
        
        train_loaders = [Drishti_train_loader, kaggle_train_loader, rim_train_loader, refuge_train_loader]
        test_loaders = [Drishti_test_loader, kaggle_test_loader, rim_test_loader, refuge_test_loader]
        unnormalized_train_datasets = [unnormalized_drishti_trainset, unnormalized_kaggle_trainset, unnormalized_rim_trainset, unnormalized_refuge_trainset]
        train_datasets = [drishti_trainset, kaggle_trainset, rim_trainset, refuge_trainset]
        test_datasets = [drishti_testset, kaggle_testset, rim_testset, refuge_testset]

        min_data_len = min(len(drishti_testset), len(kaggle_testset), len(rim_testset), len(refuge_testset))

    elif args.dataset == 'cifar10c':

        MEANS = [[0, 0, 0] for _ in range(57)]
        STDS = [[1, 1, 1] for _ in range(57)]

        transform_unnormalized = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor()
        ])
        
        train_datasets = []
        test_datasets = []
        train_loaders = []
        test_loaders = []
        for i in range(57):
            trainset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed', train=True, client_num = i, transform=transform_unnormalized)
            testset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed', train=False, client_num = i, transform=transform_unnormalized)
            train_loader_tmp = torch.utils.data.DataLoader(trainset_tmp, batch_size=args.batch, shuffle=True)
            test_loader_tmp = torch.utils.data.DataLoader(testset_tmp, batch_size=args.batch, shuffle=False)
            train_datasets.append(trainset_tmp)
            test_datasets.append(testset_tmp)
            train_loaders.append(train_loader_tmp)
            test_loaders.append(test_loader_tmp)

        min_data_len = min([len(test_dataset) for test_dataset in test_datasets])

    elif args.dataset == 'cifar10c_alpha1':

        MEANS = [[0, 0, 0] for _ in range(57)]
        STDS = [[1, 1, 1] for _ in range(57)]

        transform_unnormalized = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor()
        ])
        
        train_datasets = []
        test_datasets = []
        train_loaders = []
        test_loaders = []
        for i in range(57):
            trainset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed_alpha1', train=True, client_num = i, transform=transform_unnormalized)
            testset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed_alpha1', train=False, client_num = i, transform=transform_unnormalized)
            train_loader_tmp = torch.utils.data.DataLoader(trainset_tmp, batch_size=args.batch, shuffle=True)
            test_loader_tmp = torch.utils.data.DataLoader(testset_tmp, batch_size=args.batch, shuffle=False)
            train_datasets.append(trainset_tmp)
            test_datasets.append(testset_tmp)
            train_loaders.append(train_loader_tmp)
            test_loaders.append(test_loader_tmp)

        min_data_len = min([len(test_dataset) for test_dataset in test_datasets])
    
    elif args.dataset == 'cifar10c_alpha5':

        MEANS = [[0, 0, 0] for _ in range(57)]
        STDS = [[1, 1, 1] for _ in range(57)]

        transform_unnormalized = transforms.Compose([
                transforms.Resize(im_size),            
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation((-30,30)),
                transforms.ToTensor()
        ])
        
        train_datasets = []
        test_datasets = []
        train_loaders = []
        test_loaders = []
        for i in range(57):
            trainset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed_alpha5', train=True, client_num = i, transform=transform_unnormalized)
            testset_tmp = CIFAR10C_preprocessed(base_path='./data/CIFAR-10-C/preprocessed_alpha5', train=False, client_num = i, transform=transform_unnormalized)
            train_loader_tmp = torch.utils.data.DataLoader(trainset_tmp, batch_size=args.batch, shuffle=True)
            test_loader_tmp = torch.utils.data.DataLoader(testset_tmp, batch_size=args.batch, shuffle=False)
            train_datasets.append(trainset_tmp)
            test_datasets.append(testset_tmp)
            train_loaders.append(train_loader_tmp)
            test_loaders.append(test_loader_tmp)

        min_data_len = min([len(test_dataset) for test_dataset in test_datasets])

    else:
        NotImplementedError


    shuffled_idxes = [list(range(0, len(test_datasets[idx]))) for idx in range(len(test_datasets))]
    for idx in range(len(shuffled_idxes)):
        random.shuffle(shuffled_idxes[idx])
    concated_test_set = [torch.utils.data.Subset(test_datasets[idx], shuffled_idxes[idx][:min_data_len]) for idx in range(len(test_datasets))]
    concated_test_set = torch.utils.data.ConcatDataset(concated_test_set)
    print(len(concated_test_set))
    concated_test_loader = torch.utils.data.DataLoader(concated_test_set, batch_size=args.batch, shuffle=False)

    return train_datasets, test_datasets, train_loaders, test_loaders, concated_test_loader, MEANS, STDS