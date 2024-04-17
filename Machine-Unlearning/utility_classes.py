import torch 
# from utils import make_forget_object
from utils import *


class ForgetObject:
    """
    Has the D_f train and the D_f test loaders and the D_r_train and the D_r_test loaders for a particular domain(single)
    """
    def __init__(self, fg_train, fg_test, rem_train, rem_test):
        self.forget_train_loader = fg_train
        self.forget_test_loader = fg_test
        self.remember_train_loader = rem_train
        self.remember_test_loader = rem_test
        pass
    
    
    
class UnlearningInstance:
    def __init__(self, args, information):
        """
        Returns the instance that contains both the source and the target infomration in complete totality
        along with their respective forget objects
        """
        
        forget_labels = [int(x) for x in args.forget.split(',')]
        
        source_index = information['dataset_names'].index(args.source)
        target_index = information['dataset_names'].index(args.target)
        
        self.source_index = source_index
        self.target_index = target_index
        
        """
        Source Loading to make source elements
        """
        #region
        self.full_source_trainset = information['trainset'][source_index]
        self.full_source_train_loader = information['train_loader'][source_index]
        
        self.full_source_testset = information['testset'][source_index]
        self.full_source_test_loader = information['test_loader'][source_index]
        
        self.source_forget = make_forget_object(self.full_source_trainset, self.full_source_testset, 
                                                forget_labels, args, 
                                                information['dataset_names'], information['num_classes'])
        
    
        self.full_source_val_loader = information['val_loader'][source_index]
        
        #endregion
        
        
        """
        Target Loading to make target elements
        """
        #region
        self.full_target_trainset = information['trainset'][target_index]
        self.full_target_train_loader = information['train_loader'][target_index]
        
        self.full_target_testset = information['testset'][target_index]
        self.full_target_test_loader = information['test_loader'][target_index]
        
        self.full_target_val_loader = information['val_loader'][target_index]

        self.target_forget = make_forget_object(self.full_target_trainset, self.full_target_testset,
                                                forget_labels, args, 
                                                information['dataset_names'], information['num_classes'])
        
        #endregion

        pass
    

def make_forget_object(trainset, testset, forget_labels, args, dataset_names, num_classes=10):
    """
    Returns the training and the test loader of the dataset that has the labelse that need to be forgotten
    can be optimized if I cna find the indices of the train and the target without iterating through the whole thing
    """
    
    # breakpoint()
    """
    Failed attempt at making this code faster
    """
    #region
    # try:
    #     train_id_forget = np.where([i in forget_labels for i in trainset.targets])[0]
    #     train_id_remember = np.where([i in remember_labels for i in trainset.targets])[0]
    #     test_id_forget = np.where([i in forget_labels for i in testset.targets])[0]
    #     test_id_remember = np.where([i in remember_labels for i in testset.targets])[0]
    # except:
    #     train_id_forget = np.where([i in forget_labels for i in trainset.labels])[0]
    #     train_id_remember = np.where([i in remember_labels for i in trainset.labels])[0]
    #     test_id_forget = np.where([i in forget_labels for i in testset.labels])[0]
    #     test_id_remember = np.where([i in remember_labels for i in testset.labels])[0]
        
    # record indices through enumeration of the data
    #endregion
    
    train_id_forget = []
    train_id_remember = []
    test_id_forget = []
    test_id_remember = []
    
    for index, (img, label) in tqdm(enumerate(trainset), desc = "Creating Forget and Remember Loaders Train"):
        if label in forget_labels:
            train_id_forget.append(index)
        else:
            train_id_remember.append(index)
    
    for index, (img, label) in tqdm(enumerate(testset), desc = "Creating Forget and Remember Loaders Test"):
        if label in forget_labels:
            test_id_forget.append(index)
        else:
            test_id_remember.append(index)
    
    if (args.verbose):
        print(f'Train Forget Length = {len(train_id_forget)}')
        print(f'Train Remember Length = {len(train_id_remember)}')
        print(f'Test Forget Length = {len(test_id_forget)}')
        print(f'Test Remember Length = {len(test_id_remember)}')
    
    train_forget_loader = DataLoader(trainset, batch_size=args.batch, sampler = torch.utils.data.SubsetRandomSampler(train_id_forget))
    train_remember_loader = DataLoader(trainset, batch_size=args.batch, sampler = torch.utils.data.SubsetRandomSampler(train_id_remember))
    test_forget_loader = DataLoader(testset, batch_size=args.batch, sampler = torch.utils.data.SubsetRandomSampler(test_id_forget))
    test_remember_loader = DataLoader(testset, batch_size=args.batch, sampler = torch.utils.data.SubsetRandomSampler(test_id_remember))
    
    
    return ForgetObject(train_forget_loader, test_forget_loader, train_remember_loader, test_remember_loader)
