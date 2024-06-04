import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
from collections import defaultdict
import random
from tqdm import tqdm
import sys

from transformers import XGLMTokenizer, XGLMForCausalLM

RETRIEVAL_PROMPT = '''I am an intelligent image captioning bot. Similar images have the following captions: '''

tokenizer = XGLMTokenizer.from_pretrained("facebook/opt-350m", cache_dir="/scratch/project/dd-23-80/cache", padding_side='left')
model = XGLMForCausalLM.from_pretrained("facebook/opt-350m", cache_dir="/scratch/project/dd-23-80/cache")


class CapDataset(Dataset):
    def __init__(self, img_cap_dict, tokenizer, k=4, n=3):
        self.img_cap_data = list(img_cap_dict.items())
        self.img_cap_dict = img_cap_dict
        self.tokenizer = tokenizer
        self.img_ids = set(img_cap_dict.keys())
        self.k = k # num of captions in prompt for each shot
        self.n = n # num of shots
        
    def build_fewshot_prompt(self, idx, k, n, use_eos_token=False):
        cur_img_id = self.img_cap_data[idx][0]
        fewshot_ex_ids = random.sample(self.img_ids-set([cur_img_id]), n)
        prompt = ""
        for ex_id in fewshot_ex_ids:
            captions = self.img_cap_dict[ex_id]
            prompt += get_prompt(captions, k, is_shot=True, use_eos_token=use_eos_token)
            
        prompt += get_prompt(self.img_cap_dict[cur_img_id], k, is_shot=False, use_eos_token=use_eos_token)
        return prompt

    def __getitem__(self, idx):
        sample_prompt = self.build_fewshot_prompt(idx, k=self.k, n=self.n, use_eos_token=True)
        cur_img_id = self.img_cap_data[idx][0]
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

def build_fewshot_prompt(img_cap_dict, cur_img_id, k, n, use_eos_token=False):
    fewshot_ex_ids = random.sample(set(img_cap_dict.keys())-set([cur_img_id]), n)
    prompt = ""
    for ex_id in fewshot_ex_ids:
        captions = img_cap_dict[ex_id]
        prompt += get_prompt(captions, k, is_shot=True, use_eos_token=use_eos_token)
        
    prompt += get_prompt(img_cap_dict[cur_img_id], k, is_shot=False, use_eos_token=use_eos_token)
    return prompt

def fewshot_generation(model, tokenizer, prompt, max_new_tokens=30, do_sample=False, top_k=50, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p, num_return_sequences=1, num_beams=5, num_beam_groups=5, diversity_penalty=1.0)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    
def build_img_cap_dict(data):
    img_cap_dict = defaultdict(list)
    for item in data:
        img_cap_dict[item["image"]].append(item["caption"])
    return img_cap_dict


    
   
    
if __name__ == "__main__":
    # read data
    
    nocaps_val = ""
    # batch_size = 2
    # input_file = "/scratch/project/dd-23-80/data/beit3_data/data/flickr30k_train_preprocessed_{}.json".format(sys.argv[1])
    # with open(input_file, "r") as input_json:
    #     data = json.load(input_json)
    
    
    # # build img_cap_dict   
    # flickr_img_cap_dict = build_img_cap_dict(data)
    
    # data_loader = torch.utils.data.DataLoader(
    #                 CapDataset(flickr_img_cap_dict, tokenizer),
    #                 batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    # ) 
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = model.to(device)
    ## tokenizer = tokenizer.to(device)
    
    # with open(input_file.replace(".json", "_new_captions_no_repeat.jsonl"), "w") as out_file:
    #     for data in tqdm(data_loader, desc="Generate with batch size: " + str(batch_size)):
    #         encoding = tokenizer(data[0], padding=True, return_tensors='pt').to(device)
    #         with torch.no_grad():
    #             generated_ids = model.generate(**encoding, 
    #                                            min_new_tokens=5, max_new_tokens=30, do_sample=False, 
    #                                            top_k=50, top_p=0.95, num_return_sequences=5, num_beams=5, early_stopping=True, 
    #                                            num_beam_groups=5, diversity_penalty=1.0, no_repeat_ngram_size=2
    #                                            )
    #             generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    #             for i in range(len(generated_texts)):
    #                 img_id_idx = i//5
    #                 new_caption = generated_texts[i].split("A creative short caption I can generate to describe this image is:")[-1].strip()
    #                 # write new caption to file
    #                 out_file.write(json.dumps({"image": data[1][img_id_idx], "caption": new_caption}) + "\n")
            
    
    new_captions = defaultdict(list)
    for cur_img_id in tqdm(flickr_img_cap_dict.keys(), total=len(flickr_img_cap_dict.keys())):
        sample_prompt = build_fewshot_prompt(flickr_img_cap_dict, cur_img_id, k=4, n=3, use_eos_token=True)
        generated = fewshot_generation(model, tokenizer, sample_prompt)
        for i in range(len(generated)):
            new_caption = generated[i].split("A creative short caption I can generate to describe this image is:")[-1].strip()
            new_captions[cur_img_id].append(new_caption)
            
    # save new captions
    with open("/scratch/project/dd-23-80/data/beit3_data/data/flickr30k_train_preprocessed_new_captions.json", "w") as output_json:
        json.dump(new_captions, output_json)
        
        
