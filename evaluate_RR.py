import json
import argparse
import numpy as np
import torch
from munch import munchify
from torch.utils.data import DataLoader

from data.indoor_data import IndoorTestDataset
from engine.trainer import EpochBasedTrainer

from models.models.igasa import IGASA
from engine.evaluator import Evaluator


parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", default='3DMatch', choices=['3DMatch', '3DLoMatch'])
parser.add_argument("--config", default='./config/3dmatch.json', type=str)
parser.add_argument("--load_pretrained", default='cast-epoch-05', type=str)
parser.add_argument("--ransac", default=False, action="store_true")

_args = parser.parse_args()

class Tester(EpochBasedTrainer):
    def __init__(self, cfg):
        super(Tester, self).__init__(cfg)
        self.val_dataset = IndoorTestDataset(
            cfg.data.root, _args.benchmark,
            cfg.data.npoints, cfg.data.voxel_size,
            cfg.data_list, True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1,
                                     num_workers=cfg.data.num_workers,
                                     shuffle=False, pin_memory=True)
        self.model = IGASA(cfg.model).cuda()
        self.evaluator = Evaluator(cfg.eval).cuda()
        self.meter_dict = {
            'RR@250': [],
            'RR@500': [],
            'RR@1000': [],
            'RR@2500': [],
            'RR@5000': []
        }

    def step(self, data_dict):
        output_dict = self.model(*data_dict[:3])
        output_dict['covariance'] = data_dict[-1][0]
        
        indices = torch.argsort(output_dict['corr_confidence'], descending=True)
        sorted_corres = output_dict['corres'][indices]
        sorted_confidence = output_dict['corr_confidence'][indices]
        
        results = {}
        for num in [250, 500, 1000, 2500, 5000]:
            temp_output = output_dict.copy()
            temp_output['corres'] = sorted_corres[:num]
            temp_output['corr_confidence'] = sorted_confidence[:num]
            rr_result = self.evaluator(temp_output)  
            results['RR@%d' % num] = rr_result.get('RR', 0)
        
        return results

if __name__ == "__main__":
    with open(_args.config, 'r') as cfg:
        args = json.load(cfg)
        args = munchify(args)
        args.model.ransac = _args.ransac
    
    tester = Tester(args)
    tester.set_eval_mode()
    tester.load_snapshot(_args.load_pretrained)
    # e.g. tester.load_snapshot("cast-epoch-05")
    result_list = tester.validate_epoch()
    # RRs = result_list.meter_dict['RR'].records
    # splits = {}
    
    # for data_dict, recall in zip(tester.val_dataset.dataset, RRs):
    #     scene = data_dict['points1'].split('/')[-3]
    #     if scene not in splits.keys():
    #         splits[scene] = []
    #     splits[scene].append(recall)
    
    # print("Registration Recalls:")
    # splits = {k:np.array(v).mean() for k,v in splits.items()}
    # for k, v in splits.items(): print(k, v)
    # print("Average Registration Recall:", np.array([v for v in splits.values()]).mean())

