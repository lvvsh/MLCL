import argparse as ag
import json

def get_parser_with_args():
    parser = ag.ArgumentParser(description='Training change detection network')


    #project save
    parser.add_argument('--project name', default='/home/hdda/liangyizhou/mySEIFNetcopy/checkpoints/LEVIR-CD_SEIFNet_ce_Adamw_0.0001_200/0.9396200369051919.txt', type=str)
    parser.add_argument('--path', default='checkpoints', type=str, help='path of saved model')
    # parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    #network
    parser.add_argument('--backbone', default='vitae', type=str, choices=['resnet','swin','vitae'], help='type of model')

    parser.add_argument('--dataset', default='levir+', type=str, choices=['cdd','levir','levir+'], help='type of dataset')

    parser.add_argument('--mode', default='rsp_100', type=str, choices=['imp','rsp_40', 'rsp_100', 'rsp_120' , 'rsp_300', 'rsp_300_sgd', 'seco'], help='type of pretrn')


    return parser

