"""
Machine Unlearning via source free domain adaptation is going to be experimented with
Baselines : Golathkar 
"""

# import argparse
from OH_Dataset import *
from utils import *
from parser_utils import *
from utility_classes import *



if __name__ == "__main__":
    args = parse_args()
    information = return_domain_information(args)
    ui = UnlearningInstance(args, information)
    
    # now we bring a model in that can handle this information
    
    pass