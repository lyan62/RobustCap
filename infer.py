import pandas as pd
import argparse
import os
from tqdm import tqdm
import json
from PIL import Image
import h5py
from PIL import ImageFile
import torch
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.modeling_outputs import BaseModelOutput
from src.vision_encoder_decoder import SmallCap

from src.utils import load_data_for_inference, load_data_for_inference_splitted, prep_strings, prep_reason_strings, postprocess_preds
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pdb
from peft import PeftModel
import clip


PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25

def evaluate_norag_model(args, feature_extractor, tokenizer, model, eval_df):
    """Models without retrival augmentation can be evaluated with a batch of length >1."""
    out = []
    bs = args.batch_size
    
    for idx in tqdm(range(0, len(eval_df), bs)):
        file_names = eval_df['file_name'][idx:idx+bs]
        image_ids = eval_df['image_id'][idx:idx+bs]
        decoder_input_ids = [prep_strings('', tokenizer, is_test=True, max_length=140) for _ in range(len(image_ids))] 
        
        # load image 
        images = [Image.open(args.images_dir + file_name).convert("RGB") for file_name in file_names]
        pixel_values = feature_extractor(images, return_tensors="pt").pixel_values
        with torch.no_grad():
            preds = model.generate(pixel_values.to(args.device), 
                            decoder_input_ids=torch.tensor(decoder_input_ids).to(args.device),
                            **args.generation_kwargs)
        preds = tokenizer.batch_decode(preds)
 
        for image_id, pred in zip(image_ids, preds):
            pred = postprocess_preds(pred, tokenizer, args)
            out.append({"image_id": int(image_id), "caption": pred})

    return out

def evaluate_ret_embed_model(args, feature_extractor, tokenizer, model, eval_df):
    out = []
    
    if args.features_path is not None:
        features = h5py.File(args.features_path, 'r')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("RN50x64", device=device, download_root="/scratch/project/dd-23-80/code/RobCap")


    for idx in tqdm(range(len(eval_df))):
        file_name = eval_df['file_name'][idx]
        image_id = eval_df['image_id'][idx]
        decoder_input_ids = prep_strings('', tokenizer, is_test=True, max_length=140) 
        caps = eval_df['caps'][idx][:args.k]
        
        image = Image.open(os.path.join(args.images_dir, file_name)).convert("RGB")
        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            input_ids = clip.tokenize(caps)
            encoded_captions = clip_model.encode_text(input_ids.to(device)).cpu().numpy() # tensor
            pred = model.generate(pixel_values.to(args.device), 
                        decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                        ret_embeds=torch.tensor([encoded_captions], dtype=torch.float32).to(args.device),
                        **args.generation_kwargs)
        # pdb.set_trace()
        pred = tokenizer.decode(pred[0])
        
        if args.robust_prompting:
            pred, reason_str = postprocess_preds(pred, tokenizer, args)
            out.append({"image_id": int(image_id), "caption": pred, "useful_cap_str": reason_str})
        else:
            pred = postprocess_preds(pred, tokenizer, args)
            out.append({"image_id": int(image_id), "caption": pred})

    return out

def evaluate_rag_model(args, feature_extractor, tokenizer, model, eval_df):
    """RAG models can only be evaluated with a batch of length 1."""
    
    template = open(args.template_path).read().strip() + ' '

    if args.features_path is not None:
        features = h5py.File(args.features_path, 'r')


    out = []
    for idx in tqdm(range(len(eval_df))):
        file_name = eval_df['file_name'][idx]
        image_id = eval_df['image_id'][idx]
        caps = eval_df['caps'][idx]
        if args.robust_prompting:
            max_length = 160 # budget for reasoning midfix
            decoder_input_ids = prep_reason_strings('', tokenizer, template=template, retrieved_caps=caps,
                                                   k=int(args.k), is_test=True, max_length=max_length, use_cap_tokens=args.use_cap_tokens)
        else:
            max_length = 140
            decoder_input_ids = prep_strings('', tokenizer, template=template, retrieved_caps=caps,
                                            k=int(args.k), is_test=True, max_length=max_length, 
                                            order=args.order,
                                            add_reg_tokens=args.add_lm_reg_tokens,
                                            num_reg_tokens=args.num_lm_reg_tokens)
        # load image
        if args.features_path is not None:
            encoder_last_hidden_state = torch.FloatTensor([features[image_id][()]])
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state.to(args.device))
            with torch.no_grad():
                pred = model.generate(encoder_outputs=encoder_outputs,
                               decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                               **args.generation_kwargs)
        else:
            image = Image.open(os.path.join(args.images_dir, file_name)).convert("RGB")
            pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
            with torch.no_grad():
                pred = model.generate(pixel_values.to(args.device),
                               decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                               **args.generation_kwargs)
        # pdb.set_trace()    
        pred = tokenizer.decode(pred[0])
        
        if args.robust_prompting:
            pred, reason_str = postprocess_preds(pred, tokenizer, args)
            out.append({"image_id": int(image_id), "caption": pred, "useful_cap_str": reason_str})
        else:
            pred = postprocess_preds(pred, tokenizer, args)
            out.append({"image_id": int(image_id), "caption": pred})

    return out

def load_model(args, checkpoint_path):
    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    if args.add_vision_reg:
        model = SmallCap.from_encoder_decoder_pretrained(
        args.encoder_name, args.decoder_name, 
        cross_attention_reduce_factor=config.decoder.cross_attention_reduce_factor,
        add_vision_reg=args.add_vision_reg)
        model.load_state_dict(torch.load(checkpoint_path + '/pytorch_model.bin'))
    else:
        # pdb.set_trace()
        model = AutoModel.from_pretrained(checkpoint_path)
        if args.adapter_name is not None:
            # we only added lora to decoder during training, so here we also only add it back here
            model.decoder = PeftModel.from_pretrained(model.decoder, args.adapter_name)
            model.decoder = model.decoder.merge_and_unload()
        # model = PeftModel.from_pretrained(SmallCap, checkpoint_path)
    # model.config = config
    model.eval()
    model.to(args.device)
    return model

def infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn):
    model = load_model(args, checkpoint_path)
    
    if args.add_lm_reg_tokens:
        special_tokens_dict = {'additional_special_tokens': ['[REG1]','[REG2]','[REG3]','[REG4]']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.decoder.resize_token_embeddings(len(tokenizer))
        
    preds = infer_fn(args, feature_extractor, tokenizer, model, eval_df)
    with open(os.path.join(checkpoint_path, args.outfile_name), 'w') as outfile:
        json.dump(preds, outfile)

def register_model_and_config():
    from transformers import AutoModelForCausalLM
    from src.vision_encoder_decoder import SmallCap, SmallCapConfig
    from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
    from src.opt import ThisOPTConfig, ThisOPTForCausalLM
    from src.xglm import ThisXGLMConfig, ThisXGLMForCausalLM

    AutoConfig.register("this_xglm", ThisXGLMConfig)
    AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
    AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)

    AutoConfig.register("this_opt", ThisOPTConfig)
    AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
    AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)
    
    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    
    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)

def main(args):

    register_model_and_config()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.infer_test or args.disable_rag:
        args.features_path = None
    
    if args.features_path is not None:
        feature_extractor = None
    else:
        feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)

    if args.disable_rag:
        args.k=0
        infer_fn = evaluate_norag_model
    elif args.use_ret_embeds:
        infer_fn = evaluate_ret_embed_model
    else:
        infer_fn = evaluate_rag_model

    if args.infer_test:
        split = 'test'
    else:
        split = 'val'
        
    if args.dataset == "vizwiz":
        data = load_data_for_inference_splitted(args.annotations_path, args.captions_path, split)
    else:
        data = load_data_for_inference(args.annotations_path, args.captions_path, split)

    eval_df = pd.DataFrame(data[split])
    #args.outfile_name = '{}_preds.json'.format(split)
    args.outfile_name = '{}_{}_preds{}.json'.format(split, args.dataset, args.outfile_postfix)


    # load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN
    
    if args.robust_prompting and args.use_cap_tokens:
        special_tokens_dict = {'additional_special_tokens': ['<pred>', '<sor>', '<eor>', '<cap0>', '<cap1>', '<cap2>', '<cap3>']} # start of reasoning, end of reasoning
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    
    # configure generation 
    args.generation_kwargs = {'max_new_tokens': CAPTION_LENGTH, 'no_repeat_ngram_size': 0, 'length_penalty': 0.,
                              'num_beams': 3, 'early_stopping': True, 'eos_token_id': tokenizer.eos_token_id}

    # run inference once if checkpoint specified else run for all checkpoints
    if args.checkpoint_path is not None:
        checkpoint_path = os.path.join(args.model_path, args.checkpoint_path)
        infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn)
    else:
        for checkpoint_path in os.listdir(args.model_path):
            if 'runs' in checkpoint_path:
                continue
            checkpoint_path = os.path.join(args.model_path, checkpoint_path)
            if os.path.exists(os.path.join(checkpoint_path, args.outfile_name)):
                print('Found existing file for', checkpoint_path)
            else:
                infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--images_dir", type=str, default="data/images/", help="Directory where input image features are stored")
    parser.add_argument("--features_path", type=str, default=None, help="H5 file with cached input image features")
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json", help="JSON file with annotations in Karpathy splits")
        
    parser.add_argument("--model_path", type=str, default=None, help="Path to model to use for inference")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")

    parser.add_argument("--infer_test", action="store_true", default=False, help="Use test data instead of val data")

    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-base-patch32", help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="gpt2", help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--disable_rag", action="store_true", default=False, help="Disable retrieval augmentation or not")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64", help="Visual encoder used for retieving captions")
    parser.add_argument("--captions_path", type=str, default="data/retrieved_caps_resnet50x64.json", help="JSON file with retrieved captions")
    parser.add_argument("--template_path", type=str, default="src/template.txt", help="TXT file with template")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size; only matter if evaluating a norag model")
    
    # additional parameters for robust-cap
    parser.add_argument("--outfile_postfix", type=str, default='', required=False,
                        help="a customized postfix to be added to the inference output file")
    parser.add_argument("--dataset", type=str, default="coco", help="Use xm3600 data instead of coco data")
    parser.add_argument("--add_lm_reg_tokens", action="store_true", default=False, help="If true add reg tokens to the tokenizer and the input of the decoder")
    parser.add_argument("--num_lm_reg_tokens", type=int, default=4, help="add reg tokens to the tokenizer and the input of the decoder")
    parser.add_argument("--robust_prompting", action="store_true", default=False, help="If true permute retrived captions during training")
    parser.add_argument("--add_vision_reg", action="store_true", default=False, help="If true add reg tokens to the encoder")
    parser.add_argument("--use_cap_tokens", action="store_true", required=False, help="add special tokens for caption idxes and reasoning")
    
    parser.add_argument("--adapter_name", type=str, default=None, help="Adapter name as found of HuggingFace or stored locally")
    parser.add_argument("--use_ret_embeds", action="store_true", default=False, help="If true use ret_embeds as input to decoder")
    
    ## order
    parser.add_argument("--order", type=str, default="default", choices=['default', 'shuffle', 'permute', 'reverse', 'topk_reverse'], 
                        help="order sensitivity, choices from ['shuffle', 'permute', 'reverse']")
    
    # parser.add_argument("--xa_first", action="store_true", default=False, help="If true put xa layer before sa")
    
    
    args = parser.parse_args()

    main(args)
   
