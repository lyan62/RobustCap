'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import pdb



def nocaps_caption_eval(gt_root, results_file, split, domain:str=""):
    # urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
    #         'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    if domain != "":
         filenames = {'val':'nocaps_val_%s.json'%domain}    
    else:
        filenames = {'val':'nocaps_val_4500_captions.json'}    
    
    # download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(gt_root,filenames[split])
    
    # create coco object and coco_result object
    # pdb.set_trace()
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval



def main(args):
    
    with open(args.all_val_result_file, "r") as result_json:
        val = json.load(result_json)
        result_json.close()
    
    # gt_path = "/home/rdp455/BLIP/annotation/nocaps_gt/nocaps_val_4500_captions.json"
    with open("/scratch/project/dd-23-80/code/RobCap/annotation/nocaps_gt/nocaps_val_4500_captions.json", "r") as input_json:
        ann = json.load(input_json)

    target_domain = args.target_domain


    # get gt
    domain_val_ids = []
    domain_val_samples = []

    for data in ann["images"]:
        if data["domain"] == target_domain:
            domain_val_ids.append(data["id"])

    for val_sample in val:
        if val_sample["image_id"] in domain_val_ids:
            domain_val_samples.append(val_sample)


    val_result_file = args.all_val_result_file.replace(".json", "_%s.json"%target_domain)
    with open(val_result_file, "w") as out_json:
        json.dump(domain_val_samples, out_json)
    
    coco_val = nocaps_caption_eval(args.gt_root, val_result_file,'val', target_domain)
    # coco_test = nocaps_caption_eval(config['nocaps_gt_root'],test_result_file,'test')
                
    log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()}}
    with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
        f.write(json.dumps(log_stats) + "\n")                   
    
        
    log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()}}
    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        f.write(json.dumps(log_stats) + "\n")     
                    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output/NoCaps')
    parser.add_argument('--gt_root', default='/scratch/project/dd-23-80/code/RobCap/annotation/nocaps_gt')        
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--target_domain', type=str, default="out-domain")
    parser.add_argument('--all_val_result_file', type=str, default="/home/rdp455/BLIP/BLIP/output/nocaps_overall/result/val.json")
    args = parser.parse_args()

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)