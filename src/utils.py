from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import h5py
import bisect
import random
from itertools import permutations
import re
import pdb
import clip

CAPTION_LENGTH = 25
SIMPLE_PREFIX = "This image shows "

def get_permute(l):
    # l: list of caps
    all_permutations = list(permutations(l))
    filtered_permutations = [p for p in all_permutations if p != tuple(l) and p != tuple(reversed(l))]
    if len(filtered_permutations) > 0:
        permuted_cap = random.choice(filtered_permutations)
    else:
        # use original list
        permuted_cap = l
    return permuted_cap

def build_infix(retrieved_caps, k, order="default", seed=42):
    if order == "sample":
        random.seed(seed)
        ## random select k caps from retrieved caps
        if len(retrieved_caps) >= k:
            shuffled_caps = random.sample(retrieved_caps, k)
            infix = '\n\n'.join(shuffled_caps) + '.'
        else:
            random.shuffle(retrieved_caps)
            infix = '\n\n'.join(retrieved_caps) + '.'   
    elif order == "c-samplek": # mixed shuffle
        assert k > 1
        if len(retrieved_caps) >= k: # mixed shuffle, 1 top + 3 random
            
            shuffled_caps = random.sample(retrieved_caps[1:], k-1)
            input_caps = [retrieved_caps[0]] + shuffled_caps
            random.shuffle(input_caps)
            infix = '\n\n'.join(input_caps) + '.'
        else:
            random.shuffle(retrieved_caps)
            infix = '\n\n'.join(retrieved_caps) + '.'   
    elif order == "permute":
        permuted_caps = get_permute(retrieved_caps[:k])
        infix = '\n\n'.join(permuted_caps) + '.'
    elif order == "reverse":
        infix = '\n\n'.join(retrieved_caps[::-1][:k]) + '.'
    elif order == "topk_reverse":
        infix = '\n\n'.join(retrieved_caps[:k][::-1]) + '.'
    else:
        infix = '\n\n'.join(retrieved_caps[:k]) + '.'
    return infix


def prep_strings(text, tokenizer, template=None, retrieved_caps=None, k=None, is_test=False, max_length=None, 
                 order="default", seed=42, drop=0):

    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True
    
    if retrieved_caps is not None:
        if k == 0:
            prefix = SIMPLE_PREFIX
        else:
            infix = build_infix(retrieved_caps, k, order=order, seed=seed)
            
            prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX

    prefix_ids = tokenizer.encode(prefix)
    if drop > 0:
        drop_num = round(len(prefix_ids[5:]) * drop) 
        drop_idx = random.sample(range(5, len(prefix_ids)-1), drop_num)
        prefix_ids = [prefix_ids[i] for i in range(len(prefix_ids)) if i not in drop_idx]
        
    len_prefix = len(prefix_ids)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
    
    # double check length
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    if len(label_ids) > max_length:
        label_ids = label_ids[:max_length]
        
    if is_test:
        return input_ids
    else:  
        return input_ids, label_ids

def prep_mixed_strings(text, tokenizer, template=None, retrieved_caps=None, irrelevant_caps=None, k=None, is_test=False, max_length=None, p=0.2):
    """
    prepare input_ids and label_ids for mixed retrieval caps
    """
    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True 
    
    if is_test:
        # build prompt and input_ids
        if retrieved_caps is not None:
            infix = '\n\n'.join(retrieved_caps[:k]) + '.' 
            prefix = template.replace('||', infix)
        else:
            prefix = SIMPLE_PREFIX
            
        prefix_ids = tokenizer.encode(prefix)
        
    else:
        # train
        if retrieved_caps is not None:
            # for each of the retrieved cap, it has a probability of p to be selected as irrelevant cap
            selected_caps = []
            if len(retrieved_caps) >= k:
                for i in range(k):
                    if random.random() > p:
                        selected_caps.append(retrieved_caps[i])
            else:
                selected_caps = retrieved_caps
            
            num_irrelevant_caps = k - len(selected_caps)
                
            # rest of the caps used will be irrelevant caps
            if len(irrelevant_caps) >= num_irrelevant_caps and num_irrelevant_caps > 0:
                selected_caps += random.sample(irrelevant_caps, num_irrelevant_caps)
            
            infix = '\n\n'.join(selected_caps) + '.' 
            prefix = template.replace('||', infix)
        else:
            prefix = SIMPLE_PREFIX
    
        prefix_ids = tokenizer.encode(prefix)
        len_prefix = len(prefix_ids)
        
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        if truncation:
            text_ids = text_ids[:CAPTION_LENGTH]
        input_ids = prefix_ids + text_ids 

        # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
        label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
        if padding:
            label_ids += [-100] * (max_length - len(label_ids))
    
        if len(label_ids) > max_length:
            label_ids = label_ids[:max_length]
    
    
    input_ids = prefix_ids 
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
    # double check length
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
                
    if is_test:
        return input_ids
    else:  
        return input_ids, label_ids


def postprocess_preds(pred, tokenizer, args):
    split_str = SIMPLE_PREFIX
    pred = pred.split(split_str)[-1]
    
    pred = pred.replace(tokenizer.pad_token, '')
    if pred.startswith(tokenizer.bos_token):
        pred = pred[len(tokenizer.bos_token):]
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-len(tokenizer.eos_token)]
    
    return pred

class TrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, 
                 rag=False, template_path=None, k=None, 
                 max_caption_length=25, order="default", seed=42, drop_token=0):
        self.df = df
        self.tokenizer = tokenizer
        self.features = h5py.File(features_path, 'r')

        if rag:
            self.template = open(template_path).read().strip() + ' '
            self.max_target_length = (max_caption_length  # target caption
                                     + max_caption_length * k # retrieved captions
                                     + len(tokenizer.encode(self.template)) # template
                                     + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                     )
            assert k is not None 
            self.k = k
        else:
            self.max_target_length = CAPTION_LENGTH
        self.rag = rag
        self.order = order
        
        # ret_embeds
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("RN50x64", device=self.device, download_root="/scratch/project/dd-23-80/code/RobCap")
            
        # ret caps order
        self.order = order
        self.seed = seed
        self.drop_token = drop_token
        

    def __len__(self):
        return len(self.df)

    def get_ret_embeddings(self, idx):
        if self.order == "sample":
            caps = random.sample(self.df['caps'][idx], self.k) # random select k caps
        else:
            caps = self.df['caps'][idx][:self.k] # topk caps
        with torch.no_grad():
            input_ids = clip.tokenize(caps)
            encoded_captions = self.clip_model.encode_text(input_ids.to(self.device)).cpu().numpy() # tensor
        return torch.tensor(encoded_captions)

        
    def __getitem__(self, idx):
        text = self.df['text'][idx]
        if self.rag: 
            caps = self.df['caps'][idx]
            # default setting
            if self.k == -1: # random
                k = random.randint(0, 4)
                order = "sample"
            else:
                k = self.k
                order = self.order
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                    retrieved_caps=caps, k=k, max_length=self.max_target_length,
                                                    order=order, seed=self.seed, drop=self.drop_token)
        else:
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, max_length=self.max_target_length, 
                                                     order=self.order, seed=self.seed)
        # load precomputed features
        encoder_outputs = self.features[self.df['cocoid'][idx]][()]
        # encoder_outputs = self.features[self.df['file_name'][idx]][()]

        encoding = {"encoder_outputs": torch.tensor(encoder_outputs), 
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}

        return encoding


def load_data_for_training(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'train': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if caps_path is not None:
            caps = retrieved_caps[str(item['cocoid'])]
        else:
            caps = None
        samples = []
        for sentence in item['sentences']:
            samples.append({'file_name': file_name, 'cocoid': str(item['cocoid']), 'caps': caps, 'text': ' '.join(sentence['tokens'])})
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'] += samples
        elif item['split'] == 'val':
            data['val'] += samples
    return data 

def load_mixed_data_for_training(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'train': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if caps_path is not None:
            caps = retrieved_caps[str(item['cocoid'])]["caps"] # read useful caps
            irrelevant_caps = retrieved_caps[str(item['cocoid'])]["irrelevant_caps"] # read irrelevant caps
        else:
            caps = None
        
        samples = []
        for sentence in item['sentences']:
            samples.append({'file_name': file_name, 'cocoid': str(item['cocoid']), 'caps': caps, 'irrelevant_caps': irrelevant_caps, 'text': ' '.join(sentence['tokens'])})
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'] += samples
        elif item['split'] == 'val':
            data['val'] += samples
    return data 
    

def load_data_for_inference(annot_path, caps_path=None, split="val"):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'test': [], 'val': []}

    for item in annotations:
        if item['split'] == split:
            file_name = item['filename'].split('_')[-1]
            if caps_path is not None:
                caps = retrieved_caps[str(item['cocoid'])]
            else:
                caps = None
            image = {'file_name': file_name, 'caps': caps, 'image_id': str(item['cocoid'])}
            if item['split'] == 'test':
                data['test'].append(image)
            elif item['split'] == 'val':
                data['val'].append(image)

    return data      

def load_data_for_inference_splitted(annot_path, caps_path=None, split="val"):
    # for vizwiz
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'test': [], 'val': []}

    for item in annotations:
        if split in annot_path:
            file_name = item['file_name']
            if caps_path is not None:
                caps = retrieved_caps[str(item['id'])]
            else:
                caps = None
            image = {'file_name': file_name, 'caps': caps, 'image_id': str(item['id'])}
            if split == 'test':
                data['test'].append(image)
            elif split == 'val':
                data['val'].append(image)
            else:
                raise NotImplementedError
    return data     