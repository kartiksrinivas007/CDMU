"""
Machine Unlearning via source free domain adaptation is going to be experimented with
Baselines : Golathkar 
"""

# import argparse
from utility_classes import *
from utils import *
from parser_utils import *
from Dataset_utils import *
from parser_utils import *
import pickle




if __name__ == "__main__":
    args = parse_args()
    print(args.resume)
    if (not args.resume):
        print("Creating a new instance")
        information = return_domain_information(args)
        ui = UnlearningInstance(args, information)
        # dump the ui instance into a checkpoint folder using parquet
        try: 
            checkpoint_object(ui, args)
        except:
            print("Could not save the checkpoint")
            exit(0)
        
    else:
        try:
            ui =load_checkpoint_object(args, "Unlearning_Instance")
            print("Loaded the checkpoint for Unlearning_Instance")
        except:
            print("Could not find Ui instance to resume, loading new instance")
            information = return_domain_information(args)
            ui = UnlearningInstance(args, information)
            checkpoint_object(ui, args)
        pass
    
    
        
    if (args.verbose):
        ui.display_variables()
    # now we bring a model in that can handle this information
    if (args.dset == "digits"):
        source_model = get_network("ConvNet", channel = 3, num_classes = 10, im_size=(32, 32))
        
        pass    
    pass