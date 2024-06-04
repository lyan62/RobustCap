import json
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
from collections import defaultdict
import random
from tqdm import tqdm
import sys

from transformers import AutoTokenizer, OPTForCausalLM
from tqdm import tqdm

## set up prompt and model

RETRIEVAL_PROMPT = '''I am an intelligent image captioning bot. Similar images have the following captions: '''

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", cache_dir="/scratch/project/dd-23-80/cache", padding_side='left')
model = OPTForCausalLM.from_pretrained("facebook/opt-350m", cache_dir="/scratch/project/dd-23-80/cache")

PAD_TOKEN = '!'
EOS_TOKEN = '.'
tokenizer.pad_token = PAD_TOKEN
tokenizer.eos_token = EOS_TOKEN

class NocapsDataset(Dataset):
    def __init__(self, nocaps_val, img_cap_dict, tokenizer, k=4, n=3):
        self.val_data = nocaps_val
        self.img_cap_dict = img_cap_dict
        self.tokenizer = tokenizer
        self.img_ids = set(img_cap_dict.keys())
        self.k = k # num of captions in prompt for each shot
        self.n = n # num of shots
        
    def build_fewshot_prompt(self, idx, k, n, use_eos_token=False):
        fewshot_prompt = build_fewshot_prompt(img_cap_dict=flickr_img_cap_dict, k=k, n=n, use_eos_token=use_eos_token) # n-shot with flickr30k examples
            
        # now add prompt for curent image
        sample_prompt = get_prompt(self.val_data[idx], k=4, is_shot=False, use_eos_token=False)
        
        return fewshot_prompt+"\n"+sample_prompt
            
    def __getitem__(self, idx):
        sample_prompt = self.build_fewshot_prompt(str(idx), k=self.k, n=self.n, use_eos_token=True)
        cur_img_id = int(idx)
        return sample_prompt, cur_img_id

    def __len__(self):
        return len(self.img_cap_dict)


def get_prompt(captions, k, is_shot=False, use_eos_token=False):
    prefix = RETRIEVAL_PROMPT
    for i in range(k):
        if use_eos_token:
            prefix+="{}" + tokenizer.eos_token
        else:
            prefix+="{}" + "\n"
    
    if is_shot:
        if use_eos_token:
            prefix+="A creative short caption I can generate to describe this image is: {}" + tokenizer.eos_token + " "
        else:
            prefix+="A creative short caption I can generate to describe this image is: {}" + "\n\n" #+ " "
        output = prefix.format(*captions[:k+1])
    else:
        if use_eos_token:
            prefix+="A creative short caption I can generate to describe this image is: "
        else:
            prefix+="A creative short caption I can generate to describe this image is: " #+ " "
        output = prefix.format(*captions[:k])
    return output

def build_fewshot_prompt(img_cap_dict, k, n, use_eos_token=False, same_dataset=False, cur_img_id=None, seed=42):
    random.seed(seed)
    if not same_dataset:
        fewshot_ex_ids = random.sample(set(img_cap_dict.keys()), n)
    else:
        fewshot_ex_ids = random.sample(set(img_cap_dict.keys())-set([cur_img_id]), n)
    
    # build prompt
    prompt = ""
    for ex_id in fewshot_ex_ids:
        captions = img_cap_dict[ex_id]
        prompt += get_prompt(captions, k, is_shot=True, use_eos_token=use_eos_token)
    
    # if using same dataset, build prompt for current image
    if same_dataset:
        prompt += get_prompt(img_cap_dict[cur_img_id], k, is_shot=False, use_eos_token=use_eos_token)   
    return prompt


def fewshot_generation(model, tokenizer, prompt, max_new_tokens=25, do_sample=False, top_k=50, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p, num_return_sequences=1, num_beams=5, num_beam_groups=5, diversity_penalty=1.0)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def build_img_cap_dict(data):
        img_cap_dict = defaultdict(list)
        for item in data:
            img_cap_dict[item["image"]].append(item["caption"])
        return img_cap_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="nocaps")
    parser.add_argument("--fewshot", action="store_true", default=True)
    args = parser.parse_args()
    
    ## if fewshot we use few-shot examples from Flickr30k
    input_file = "/scratch/project/dd-23-80/data/beit3_data/data/flickr30k_train_preprocessed_0.json"
    with open(input_file, "r") as input_json:
        data = json.load(input_json)

    flickr_img_cap_dict = build_img_cap_dict(data)
        
    # for nocaps  
    if args.data == "nocaps":
        preds = []
        with open("/scratch/project/dd-23-80/code/RobCap/data/nocaps.json", "r") as input_json:
            ret = json.load(input_json) 
        
        
        data_loader = torch.utils.data.DataLoader(NocapsDataset(ret, flickr_img_cap_dict, tokenizer),
                                                  batch_size=4, shuffle=False, num_workers=0, pin_memory=True) 
        
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        # tokenizer = tokenizer.to(device)
        
            
        # preds = []
        # for data in tqdm(data_loader, desc="Generate with batch size: " + str(4)):
        #     encoding = tokenizer(data[0], padding=True, return_tensors='pt').to(device)
        #     with torch.no_grad():
        #         generated_ids = model.generate(**encoding, min_new_tokens=5, max_new_tokens=25, num_beams=3, no_repeat_ngram_size=2, top_k=50, top_p=0.95, 
        #                                        num_beam_groups=3, early_stopping=True, diversity_penalty=1.0, num_return_sequences=1)
        #         output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        #         for i in range(len(output)):
        #             preds.append({"image_id": int(data[1][i]),"caption": output[i].split("A creative short caption I can generate to describe this image is:")[-1].strip().split("\n")[0]})
        # with open("/scratch/project/dd-23-80/code/RobCap/output/ra_opt/nocaps_val_preds_k4_n3.json", "w") as output_json:
        #     json.dump(preds, output_json)
    
        
        preds = []
        for img_id, ret in tqdm(ret.items(), total=len(ret)): 
            # build fewshot prompt from flickr
            fewshot_prompt = build_fewshot_prompt(img_cap_dict=flickr_img_cap_dict, k=4, n=3, use_eos_token=False) # n-shot with flickr30k examples
            
            # now add prompt for curent image
            sample_prompt = get_prompt(ret, k=4, is_shot=False, use_eos_token=False)
            
            fewshot_inputs = tokenizer(fewshot_prompt+"\n"+sample_prompt, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(fewshot_inputs, max_new_tokens=25, num_beams=3, repetitaion_penalty=1.0, no_repeat_ngram_size=2)
            output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            preds.append(
                {"image_id": int(img_id),
                 "caption": output.split("A creative short caption I can generate to describe this image is:")[-1].strip().split("\n")[0]})
        
        with open("/scratch/project/dd-23-80/code/RobCap/output/ra_opt/nocaps_val_preds_k4_n3_b1.json", "w") as output_json:
            json.dump(preds, output_json)
          