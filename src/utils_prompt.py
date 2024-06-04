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


def prep_strings(text, tokenizer, template=None, retrieved_caps=None, k=None, is_test=False, max_length=None, 
                 reverse=False, permute=False, add_reg_tokens=False, num_reg_tokens=4):

    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True
    
    if retrieved_caps is not None:
        if permute:
            permuted_caps = get_permute(retrieved_caps[:k])
            infix = '\n\n'.join(permuted_caps) + '.'
        elif reverse:
            infix = '\n\n'.join(retrieved_caps[:k][::-1]) + '.'
        else:
            infix = '\n\n'.join(retrieved_caps[:k]) + '.'
        
        if add_reg_tokens:
            for i in range(num_reg_tokens):
                reg_prefix = "".join(tokenizer.special_tokens_map["additional_special_tokens"][:i+1])
            prefix = reg_prefix + template.replace('||', infix)
        else:
            prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX

    prefix_ids = tokenizer.encode(prefix)
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




def prep_mixed_strings(text, tokenizer, template=None, retrieved_caps=None, irrelevant_caps=None, k=None, is_test=False, max_length=None):
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
            # for each of the retrieved cap, it has a probability of 0.5 to be selected
            selected_caps = []
            if len(retrieved_caps) >= k:
                for i in range(k):
                    if random.random() < 0.5:
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
    
    
def create_reason_prefix(cap_list, template):
    infix = '\n\n'.join(cap_list) + '.' 
    prefix = template.replace('||', infix)
    return prefix

def add_cap_idx(cap_list, use_cap_tokens=False):
    if use_cap_tokens:
        indexed_caps_list = ["<cap" + str(idx) + ">" + cap for (idx, cap) in enumerate(cap_list)]
    else:
        indexed_caps_list = ["(" + str(idx) + ") " + cap for (idx, cap) in enumerate(cap_list)]
    return indexed_caps_list

def create_reason_str(caps_indices, use_cap_tokens=False):
    if use_cap_tokens:
        reason_str = '<sor>' + ' '.join(['<cap' + str(cap_idx) + '>' for cap_idx in caps_indices]) + '<eor>'
    else:
        reason_str = ','.join(['(' + str(cap_idx) + ')' for cap_idx in caps_indices]) + '.'
    return reason_str


def prep_reason_strings(text, tokenizer, template=None, retrieved_caps=None, irrelevant_caps=None, k=None, 
                        is_test=False, max_length=None, use_cap_tokens=False):
    """
    prepare input_ids and label_ids for mixed retrieval caps with reasoning intermediate prompt
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
            indexed_caps_list = add_cap_idx(retrieved_caps[:k], use_cap_tokens=use_cap_tokens)
            prefix = create_reason_prefix(indexed_caps_list, template)
        else:
            prefix = SIMPLE_PREFIX
        prefix_ids = tokenizer.encode(prefix)
        
    else:
        # train
        if retrieved_caps is not None:
            ## retrieved cap order wise manipulation
            max_irr_caps = random.randint(1, k // 2)  # number of irrelevant caps not exceed k/2

            if len(retrieved_caps) >= k and len(irrelevant_caps) >= max_irr_caps:
                    # random select indexes for irrelevant caps
                irr_indices = random.sample(range(k), max_irr_caps)
                caps_indices = [idx for idx in range(k) if idx not in irr_indices]
                
                provided_caps_list = retrieved_caps[:k]
                # replace the relevant caps with irrelevant caps
                for idx, irr_idx  in enumerate(irr_indices):
                    provided_caps_list[irr_idx] = irrelevant_caps[idx]
                
            else:
                # if the number of retrieved caps or irrelevant caps is not enough, we use the original caps
                provided_caps_list = retrieved_caps[:k]
                caps_indices = list(range(len(provided_caps_list)))
            
            # build prompt
            #["(" + str(idx) + ") " + cap for (idx, cap) in enumerate(provided_caps_list)]
            indexed_caps_list = add_cap_idx(provided_caps_list, use_cap_tokens=use_cap_tokens)
            prefix = create_reason_prefix(indexed_caps_list, template)
            
            # build reason string    
            reason_str = create_reason_str(caps_indices, use_cap_tokens=use_cap_tokens) #','.join(['(' + str(cap_idx) + ')' for cap_idx in caps_indices]) + '.'
            reason_ids = tokenizer.encode(reason_str, add_special_tokens=False) # indexes of the useful retrieved caps
            max_length += 15 
        else:
            prefix = SIMPLE_PREFIX

        prefix_ids = tokenizer.encode(prefix)
        len_prefix = len(prefix_ids)
        
        # pdb.set_trace()
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        if truncation:
            text_ids = text_ids[:CAPTION_LENGTH]
        
        if retrieved_caps is not None:
            # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
            label_ids = [-100] * (len_prefix - 1) + reason_ids + text_ids + [tokenizer.eos_token_id]   
        else:
            label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
        
        if padding:
            label_ids += [-100] * (max_length - len(label_ids))
    
        if len(label_ids) > max_length:
            label_ids = label_ids[:max_length]
    
    # debug
    # print("prefix: ", prefix, "reason str: ", reason_str, "gt: ", text)
    # print("prefix ids: ", prefix_ids, "reason ids: ", reason_ids, "gt ids: ", text_ids)
    
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
    if args.robust_prompting:
        if args.use_cap_tokens:
            splitted_pred = re.split("<sor>|<eor>", pred)
            if len(splitted_pred) == 3:
                reason_str = splitted_pred[-2].strip()
                pred = splitted_pred[-1].strip()
            else:
                pred = pred.split(">")[-1].strip()
                reason_str = pred.split("The useful captions are")[-1].rstrip(pred)
        else:
            split_str = "The useful captions are"
            pred = pred.split(split_str)[-1].strip()
            try:
                splitted = pred.split(").")
                pred = splitted[-1]
                reason_str = ").".join([x for x in splitted[:-1] if x !=[""]]) \
                    .lstrip(tokenizer.bos_token) \
                    .rstrip(tokenizer.eos_token)
            except:
                pred = pred.strip()
                reason_str = ""
    else:
        split_str = SIMPLE_PREFIX
        pred = pred.split(split_str)[-1]
    # print(pred)
    pred = pred.replace(tokenizer.pad_token, '')
    if pred.startswith(tokenizer.bos_token):
        pred = pred[len(tokenizer.bos_token):]
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-len(tokenizer.eos_token)]
    
    if args.robust_prompting:
        return pred, reason_str
    else:
        return pred

class TrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, 
                 rag=False, template_path=None, k=None, 
                 max_caption_length=25, reverse=False, 
                 permute=False, add_reg_tokens=False, 
                 num_reg_tokens=4, robust_prompting=False, 
                 adversarial_training=False, use_cap_tokens=False):
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
        self.rag = rag
        self.reverse = reverse
        self.permute = permute
        self.add_reg_tokens = add_reg_tokens
        self.num_reg_tokens = num_reg_tokens
        self.robust_prompting = robust_prompting
        self.adversarial_training = adversarial_training
        self.use_cap_tokens = use_cap_tokens

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'][idx]
        if self.rag: 
            caps = self.df['caps'][idx]
            if self.robust_prompting: 
                # pdb.set_trace()
                irrelevant_caps = self.df['irrelevant_caps'][idx]
                decoder_input_ids, labels = prep_reason_strings(text, self.tokenizer, template=self.template,
                                                               retrieved_caps=caps, irrelevant_caps=irrelevant_caps, 
                                                               k=self.k, max_length=self.max_target_length, use_cap_tokens=self.use_cap_tokens)

        else:
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, max_length=self.max_target_length, 
                                                     reverse=self.reverse, permute=self.permute, 
                                                     add_reg_tokens=self.add_reg_tokens,
                                                     num_reg_tokens=self.num_reg_tokens)
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

