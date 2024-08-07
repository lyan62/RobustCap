import pandas as pd
import numpy as np
import os
import argparse
import pdb


from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM
from transformers import Seq2SeqTrainer, default_data_collator, Seq2SeqTrainingArguments

from transformers import VisionEncoderDecoderModel, CLIPModel, CLIPVisionModel,EncoderDecoderModel
from src.vision_encoder_decoder import SmallCap, SmallCapConfig
from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
from src.xglm import ThisXGLMConfig, ThisXGLMForCausalLM
from src.opt import ThisOPTConfig, ThisOPTForCausalLM

from src.utils import *
from peft import LoraConfig, get_peft_model

import os
# os.environ["WANDB_PROJECT"] = "robcap-k"
os.environ["WANDB_DISABLED"] = "true"

# for attention with 28M params, we devide the attention dimensions by 1
# for attention with 14M params, we devide the attention dimensions by 2, etc.
PARAMS2REDUCE_FACTOR = {28: 1, 14: 2, 7: 4, 3.5: 8, 1.75: 16}
PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25

def load_model(args, checkpoint_path):
    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    if args.add_vision_reg:
        model = SmallCap.from_encoder_decoder_pretrained(
        args.encoder_name, args.decoder_name, 
        cross_attention_reduce_factor=config.decoder.cross_attention_reduce_factor,
        add_vision_reg=args.add_vision_reg)
        model.load_state_dict(torch.load(checkpoint_path + '/pytorch_model.bin'))
        # pdb.set_trace()
    else:
        model = AutoModel.from_pretrained(checkpoint_path)
    # model.config = config
    # model.to(args.device)
    return model

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params_names = []
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_params_names.append(_)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    print("trainable params: \n", trainable_params_names)

def get_model_and_auxiliaries(args):

    # register model types
    if "xglm" in args.decoder_name:
        AutoConfig.register("this_xglm", ThisXGLMConfig)
        AutoModel.register(ThisXGLMConfig, ThisXGLMForCausalLM)
        AutoModelForCausalLM.register(ThisXGLMConfig, ThisXGLMForCausalLM)

    elif "opt" in args.decoder_name:
        AutoConfig.register("this_opt", ThisOPTConfig)
        AutoModel.register(ThisOPTConfig, ThisOPTForCausalLM)
        AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)

    else:
        AutoConfig.register("this_gpt2", ThisGPT2Config)
        AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
        AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    
    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)

    # create and configure model
    cross_attention_reduce_factor = PARAMS2REDUCE_FACTOR[args.attention_size]

    feature_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN
    
    # pdb.set_trace()
    # if args.continue_pretrain:
    #     model = load_model(args, args.checkpoint_path)
    # else:
    if args.resume_from_checkpoint and args.adapter_name is not None:
        model = load_model(args, args.resume_from_checkpoint)
        if "lora" in args.adapter_name:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["fc1", "fc2"], #["self_attn.q_proj", "self_attn.v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM")
        print("applying LoRA to decoder...")
        model.decoder = get_peft_model(model.decoder, lora_config)
    else:
        model = SmallCap.from_encoder_decoder_pretrained(
            args.encoder_name, args.decoder_name, 
            cross_attention_reduce_factor=cross_attention_reduce_factor,
            add_vision_reg=args.add_vision_reg,
            adapter_name=args.adapter_name,
            add_selection_layer=args.add_selection_layer,
            ret_mlp=args.use_ret_embeds,
            xa_first=args.xa_first)
        # pdb.set_trace()
    if args.add_vision_reg:
        print(model)
        
    
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder_start_token_id = None
    model.config.pad_token_id = tokenizer.pad_token_id 
    model.config.eos_token_id = tokenizer.eos_token_id 
    
    if args.add_lm_reg_tokens:
        special_tokens_dict = {'additional_special_tokens': ['[REG1]','[REG2]','[REG3]','[REG4]']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.decoder.resize_token_embeddings(len(tokenizer))
        
    if args.robust_prompting and args.use_cap_tokens:
        special_tokens_dict = {'additional_special_tokens': ['<sor>', '<eor>', '<cap0>', '<cap1>', '<cap2>', '<cap3>']} # start of reasoning, end of reasoning
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        # print(len(tokenizer))
        # if not args.resume_from_checkpoint:
        #     model.encoder.resize_token_embeddings(len(tokenizer))
        model.decoder.resize_token_embeddings(model.decoder.model.decoder.embed_tokens.weight.shape[0] + num_added_toks)
        # if "2.7b" in args.decoder_name:
        #     model.decoder.resize_token_embeddings(50279)
        # model.decoder.resize_token_embeddings(len(tokenizer))
            

    if not args.disable_rag:
        model.config.k = args.k
        model.config.retrieval_encoder = args.retrieval_encoder   
    model.config.max_length = CAPTION_LENGTH   
    model.config.rag = not args.disable_rag
    
    # pdb.set_trace()
    print("with out freezing or adapter....")
    print_trainable_parameters(model)
    print("*"*100)
    #print("model",model)
    #print(stop)
    # freeze parameters
    if args.add_vision_reg:
        for name, param in model.encoder.named_parameters():
            if name not in ['embeddings.reg_embedding', 'embeddings.reg_position_embedding.weight'] :
                param.requires_grad = False
    else:
        for param in model.encoder.parameters():
            param.requires_grad = False

    if "xglm" in args.decoder_name or "opt" in args.decoder_name:
        if not args.train_decoder:
            if args.adapter_name is not None: # update adapters
                update_xattn = args.update_xattn
                for name, param in model.decoder.named_parameters():
                    adapter_condition = args.adapter_name in name
                    encoder_attn_condition = "encoder_attn" in name
                    if adapter_condition or (update_xattn and encoder_attn_condition):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif args.add_selection_layer:
                for name, param in model.decoder.named_parameters():
                    if ('encoder_attn' not in name) and "poly_code_embeddings" not in name:
                        param.requires_grad = False
            elif args.use_ret_embeds:
                for name, param in model.decoder.named_parameters():
                    if 'encoder_attn' in name or "ret_mlp" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False    
            else:
                for name, param in model.decoder.named_parameters():
                    if 'encoder_attn' not in name:
                        param.requires_grad = False

    else:
        if not args.train_decoder:
            for name, param in model.decoder.named_parameters():
                if 'crossattention' not in name:
                    param.requires_grad = False
    # pdb.set_trace()
    # count trainable parameters
    print("after freezing/adapter....")
    print("*"*100)
    print_trainable_parameters(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Training a model with {} trainable parameters.'.format(num_trainable_params))

    return model, tokenizer, feature_extractor

def get_data(tokenizer, max_length, args):
    if args.robust_prompting or args.adversarial_training:
        data = load_mixed_data_for_training(args.annotations_path, args.captions_path)
    else:
        data = load_data_for_training(args.annotations_path, args.captions_path)
    
    train_df = pd.DataFrame(data['train'])

    if args.ablation_visual:
        train_dataset =  AblationFeaturesDataset(
                            df=train_df,
                            features_path=os.path.join(args.features_dir,'train.hdf5'),
                            tokenizer=tokenizer,
                            rag=not args.disable_rag,
                            template_path=args.template_path,
                            k=args.k,
                            max_caption_length=max_length)
    else:
        train_dataset = TrainDataset(
                            df=train_df,
                            features_path=os.path.join(args.features_dir,'train.hdf5'),
                            tokenizer=tokenizer,
                            rag=not args.disable_rag,
                            template_path=args.template_path,
                            k=args.k,
                            max_caption_length=max_length,
                            add_reg_tokens=args.add_lm_reg_tokens,
                            num_reg_tokens=args.num_lm_reg_tokens,
                            robust_prompting=args.robust_prompting,
                            adversarial_training=args.adversarial_training,
                            use_cap_tokens=args.use_cap_tokens,
                            in_prepare=args.in_prepare,
                            p=args.p, use_ret_embeds=args.use_ret_embeds,
                            order=args.order, seed=args.seed, drop_token=args.drop_token)

    return train_dataset

def main(args):    
    # torch.multiprocessing.set_start_method('spawn')
    model, tokenizer, feature_extractor = get_model_and_auxiliaries(args)
    train_dataset = get_data(tokenizer, model.config.max_length, args)
    # pdb.set_trace()
    model_type = 'norag' if args.disable_rag else 'rag'
    if args.ablation_visual:
        output_dir = '{}_{}M_{}_ablation'.format(model_type, args.attention_size, args.decoder_name)
    else:
        output_dir = '{}_{}M_{}_{}'.format(model_type, args.attention_size, args.decoder_name, args.order)

    output_dir = os.path.join(args.experiments_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=args.n_epochs, 
        per_device_train_batch_size=args.batch_size, 
        gradient_accumulation_steps=args.gradient_steps,
        learning_rate = args.lr,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=args.n_epochs, 
        logging_strategy="epoch", 
        output_dir=output_dir, 
        overwrite_output_dir=True, 
        # report_to="wandb",
        # dataloader_num_workers=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator, 
        train_dataset=train_dataset,
        tokenizer=feature_extractor,
    )
    if args.resume_from_checkpoint and args.adapter_name is not None:
        trainer.train()
    else:
        trainer.train(args.resume_from_checkpoint)
    
    ## save adapter after trained
    if args.adapter_name is not None:
        model.decoder.save_pretrained(os.path.join(output_dir, "decoder_sa_lora"), save_adapter=True, save_config=True)
        # merge lora to model, save again as a whole
        model.decoder = model.decoder.merge_and_unload()
        trainer.save_model(os.path.join(output_dir, "encoder_decoder_sa_lora_latest"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--features_dir", type=str, default="features/", help="Directory where cached input image features are stored")
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json", help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--experiments_dir", type=str, default="experiments/", help="Directory where trained models will be saved")

    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-base-patch32", help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="gpt2", help="Decoder name as found of HuggingFace or stored locally")
    parser.add_argument("--attention_size", type=float, default=7, help="Number of parameters in the cross attention {28, 14, 7, 3.5, 1.75}")
    parser.add_argument("--train_decoder", action="store_true", default=False, help="Whether to train the decoder in addition to the attention")

    parser.add_argument("--disable_rag", action="store_true", default=False, help="Disable retrieval augmentation")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64", help="Visual encoder used for retieving captions")
    parser.add_argument("--captions_path", type=str, default="data/retrieved_caps_resnet50x64.json", help="JSON file with retrieved captions")
    parser.add_argument("--template_path", type=str, default="src/template.txt", help="TXT file with template")

    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient accumulation steps")

    parser.add_argument("--ablation_visual", action="store_true", default=False, help="Whether to blank visual features")
    
    # order sensitivity, choices from ["random", "forward", "backward", "reverse"]
    parser.add_argument("--order", type=str, default="default", choices=['default', 'shuffle', 'permute', 'reverse', 'm-shuffle', 'topk_reverse'], 
                        help="order sensitivity, choices from ['shuffle', 'permute', 'reverse']")
    
    # robust training methods
    parser.add_argument("--robust_prompting", action="store_true", default=False, help="If add reasoning step training")
    parser.add_argument("--adversarial_training", action="store_true", default=False, help="If mix in irrelevant caps during training")
    
    # reg tokens
    parser.add_argument("--add_lm_reg_tokens", action="store_true", default=False, help="If true add reg tokens to the tokenizer and the input of the decoder")
    parser.add_argument("--num_lm_reg_tokens", type=int, default=4, help="add reg tokens to the tokenizer and the input of the decoder")
    parser.add_argument("--add_vision_reg", action="store_true", default=False, help="If true add reg tokens to the encoder")
    parser.add_argument("--continue_pretrain", action="store_true", default=False, help="If true add reg tokens to the encoder")
    parser.add_argument("--checkpoint_path", required=False, help="checkpoint_path for continue_pretrain")
    parser.add_argument("--resume_from_checkpoint", required=False, help="checkpoint_path for continue_pretrain")
    parser.add_argument("--use_cap_tokens", action="store_true", required=False, help="add special tokens for caption idxes and reasoning")
    parser.add_argument("--in_prepare", action="store_true", required=False, help="no reasoning step in prepare")
    parser.add_argument("--p", type=float, required=False, default=0.2, help="ratio of data to be replaced with irrelevant caps")
    
    # adapter
    parser.add_argument("--adapter_name", type=str, required=False, default=None, help="adapter name: e.g. ybelkada/opt-350m-lora")
    parser.add_argument("--update_xattn", type=bool, default=True, help="If true update xattn")
    parser.add_argument("--add_selection_layer", action="store_true", default=False, help="If true use ploy_m to filter input captions")
    parser.add_argument("--use_ret_embeds", action="store_true", default=False, help="If true use ret_embeds as input to decoder")
    parser.add_argument("--xa_first", action="store_true", default=False, help="If true put xa layer before sa")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    parser.add_argument("--drop_token", type=float, default=0.0, help="If >0  drop token in the ret caps")
    
    args = parser.parse_args()

    main(args)
