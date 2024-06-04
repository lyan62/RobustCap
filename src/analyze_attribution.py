from torch.nn import functional as F
import transformers

import pandas as pd
import argparse
import os
from tqdm import tqdm
import json
from PIL import Image, ImageFile
import h5py
import torch
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.modeling_outputs import BaseModelOutput
from pathlib import Path
import sys

from src.utils import load_data_for_inference, prep_strings, postprocess_preds
from transformers.models.auto.configuration_auto import AutoConfig
from src.vision_encoder_decoder import SmallCap, SmallCapConfig

import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from functools import partial
from typing import Any, Dict
from captum.attr import (
    IntegratedGradients,
    Saliency,
    InputXGradient,
    DeepLift,
    DeepLiftShap,
    GuidedBackprop,
    GuidedGradCam,
    Deconvolution,
    LayerGradientXActivation,
    LRP
)

from captum.attr import IntegratedGradients
from matplotlib.colors import LinearSegmentedColormap

from captum.attr._utils.input_layer_wrapper import ModelInputWrapper
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature, TextFeature
from captum.attr import TokenReferenceBase, configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import pickle


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
            pred = postprocess_preds(pred, tokenizer)
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
        decoder_input_ids = prep_strings('', tokenizer, template=template, retrieved_caps=caps,
                                                 k=int(args.k), is_test=True, max_length=140, reverse=args.reverse)
        # load image
        if args.features_path is not None:
            encoder_last_hidden_state = torch.FloatTensor([features[image_id][()]])
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state.to(args.device))
            with torch.no_grad():
                pred = model.generate(encoder_outputs=encoder_outputs,
                               decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                               **args.generation_kwargs)
        else:
            image = Image.open(args.images_dir + file_name).convert("RGB")
            pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
            with torch.no_grad():
                pred = model.generate(pixel_values.to(args.device),
                               decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                               **args.generation_kwargs)
        pred = tokenizer.decode(pred[0])

        pred = postprocess_preds(pred, tokenizer)
        out.append({"image_id": int(image_id), "caption": pred})

    return out

def load_model(args, checkpoint_path):
    config = AutoConfig.from_pretrained(checkpoint_path + '/config.json')
    model = AutoModel.from_pretrained(checkpoint_path)
    model.config = config
    model.eval()
    model.to(args.device)
    return model

def infer_one_checkpoint(args, feature_extractor, tokenizer, checkpoint_path, eval_df, infer_fn):
    model = load_model(args, checkpoint_path)
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


def format_special_chars(tokens):
    return [
        t.replace("Ġ", " ")
        .replace("▁", " ")
        .replace("</w>", "")
        .replace("ĊĊ", " ")
        .replace("Ċ", " ")
        for t in tokens
    ]
    
def predid2tokens(pred, decoder_input_len, tokenizer):
    pred_len = len(pred["sequences"][0])-decoder_input_len
    pred_tokens = format_special_chars(tokenizer.convert_ids_to_tokens(pred["sequences"][0][-pred_len:]))
    
    decoder_input_tokens = format_special_chars(tokenizer.convert_ids_to_tokens(pred["sequences"][0][:decoder_input_len]))
    return decoder_input_tokens, pred_tokens


# Cross attention visualization
def get_image_patches(eval_df, idx, feature_extractor, args):
    file_name = eval_df['file_name'][idx]
    image_id = eval_df['image_id'][idx]
    caps = eval_df['caps'][idx]

    # load image 
    image = Image.open(os.path.join(dir_path, args.images_dir + file_name)).convert("RGB")
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values

    # print(pixel_values.size())
    patch_size = (32, 32)

    # Calculate the number of patches in each dimension
    num_patches_h = pixel_values.shape[2] // patch_size[0]
    num_patches_w = pixel_values.shape[3] // patch_size[1]

    # Use the unfold function to split the image into patches
    unfolded_image = pixel_values.unfold(2, patch_size[0], patch_size[0]).unfold(3, patch_size[1], patch_size[1])

    # The shape of unfolded_image will be (1, 3, num_patches_h, num_patches_w, 32, 32)

    # Reshape the unfolded image to (num_patches_h * num_patches_w, 3, 32, 32)
    unfolded_image = unfolded_image.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 3, patch_size[0], patch_size[1])
    
    image_patches = []
    for img in unfolded_image:
        image_patches.append(img.permute(1, 2, 0).numpy())
        
    return image_patches, image_id, caps, image

def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False

def show_image_list(list_images, image_id, list_titles=None, list_cmaps=None, grid=True, num_cols=7, figsize=(10, 10), title_fontsize=5):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.
    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Patch %d' % (i+1)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)
        list_axes[i].axis("off")

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)
        

    fig.tight_layout()
    # _ = plt.show()
    # print("image_id: ", image_id)
    fig.savefig(f"{image_id}_patches.png")


ATTR_NAME_ALIASES = {
    'ig': 'integrated_gradients',
    'saliency': 'gradient',
    'dl': 'deep_lift',
    'dls': 'deep_lift_shap',
    'gb': 'guided_backprop',
    'gg': 'guided_gradcam',
    'deconv': 'deconvolution',
    'lrp': 'layer_relevance_propagation'
}

ATTR_NAME_TO_CLASS = { # TODO: Add more Captum Primary attributions with needed computed arguments
    'integrated_gradients': IntegratedGradients,
    'gradient': Saliency,
    'grad_x_input': InputXGradient,
    'deep_lift': DeepLift,
    'deep_lift_shap': DeepLiftShap,
    'guided_backprop': GuidedBackprop,
    'guided_gradcam': GuidedGradCam,
    'deconvolution': Deconvolution,
    'layer_relevance_propagation': LRP
}

def normalize_attributes(attributes: torch.Tensor) -> torch.Tensor:
    # attributes has shape (batch, sequence size, embedding dim)
    attributes = attributes.squeeze(0)

    # norm calculates a scalar value (L2 Norm)
    norm = torch.norm(attributes, dim=1)
    attributes = norm / torch.sum(norm)  # Normalize the values so they add up to 1

    return attributes
    

def compute_primary_attributions_scores(attr_method : str, model: transformers.PreTrainedModel,
                                        forward_kwargs: Dict[str, Any], prediction_id: torch.Tensor,
                                        aggregation: str = "L2") -> torch.Tensor:
    """
    Computes the primary attributions with respect to the specified `prediction_id`.

    Args:
        attr_method: Name of the primary attribution method to compute
        model: HuggingFace Transformers Pytorch language model.
        forward_kwargs: contains all the inputs that are passed to `model` in the forward pass
        prediction_id: Target Id. The Integrated Gradients will be computed with respect to it.
        aggregation: Aggregation/normalzation method to perform to the Integrated Gradients attributions.
         Currently only "L2" is implemented

    Returns: a tensor of the normalized attributions with shape (input sequence size,)

    """

    def model_forward(input_: torch.Tensor, decoder_: torch.Tensor, model, extra_forward_args: Dict[str, Any]) \
            -> torch.Tensor:
        if decoder_ is not None:
            output = model(pixel_values=input_, decoder_inputs_embeds=decoder_, **extra_forward_args)
        else:
            output = model(inputs_embeds=input_, **extra_forward_args)
        return F.softmax(output.logits[:, -1, :], dim=-1)


    extra_forward_args = {k: v for k, v in forward_kwargs.items() if
                          k not in ['inputs_embeds', 'decoder_inputs_embeds']}
    input_ = forward_kwargs.get('inputs_embeds')
    decoder_ = forward_kwargs.get('decoder_inputs_embeds')

    if decoder_ is None:
        forward_func = partial(model_forward, decoder_=decoder_, model=model, extra_forward_args=extra_forward_args)
        inputs = input_
    else:
        forward_func = partial(model_forward, model=model, extra_forward_args=extra_forward_args)
        inputs = tuple([input_, decoder_])

    attr_method_class = ATTR_NAME_TO_CLASS.get(ATTR_NAME_ALIASES.get(attr_method, attr_method), None)
    if attr_method_class is None:
        raise NotImplementedError(
            f"No implementation found for primary attribution method '{attr_method}'. "
            f"Please choose one of the methods: {list(ATTR_NAME_TO_CLASS.keys())}"
        )

    ig = attr_method_class(forward_func=forward_func)
    attributions = ig.attribute(inputs, target=prediction_id, n_steps=30, baselines=tuple([input_*0, decoder_*0]), return_convergence_delta=True)[0]

    # if decoder_ is not None:
    #     # Does it make sense to concatenate encoder and decoder attributions before normalization?
    #     # We assume that the encoder/decoder embeddings are the same
    #     return normalize_attributes(torch.cat(attributions, dim=1))
    # else:
    return attributions #normalize_attributes(attributions)



def compute_layer_attributions_scores(model: transformers.PreTrainedModel, layer,
                                      forward_kwargs: Dict[str, Any], 
                                      prediction_id: torch.Tensor) -> torch.Tensor:
    """
    Computes the primary attributions with respect to the specified `prediction_id`.

    Args:
        attr_method: Name of the primary attribution method to compute
        model: HuggingFace Transformers Pytorch language model.
        forward_kwargs: contains all the inputs that are passed to `model` in the forward pass
        prediction_id: Target Id. The Integrated Gradients will be computed with respect to it.
        aggregation: Aggregation/normalzation method to perform to the Integrated Gradients attributions.
         Currently only "L2" is implemented

    Returns: a tensor of the normalized attributions with shape (input sequence size,)

    """

    def model_forward(input_: torch.Tensor, decoder_: torch.Tensor, model, extra_forward_args: Dict[str, Any]) \
            -> torch.Tensor:
        if decoder_ is not None:
            output = model(pixel_values=input_, decoder_inputs_embeds=decoder_, **extra_forward_args)
            print(output.keys())
        else:
            output = model(pixel_values=input_, **extra_forward_args)
        return F.softmax(output.logits[:, -1, :], dim=-1)


    extra_forward_args = {k: v for k, v in forward_kwargs.items() if
                          k not in ['inputs_embeds', 'decoder_inputs_embeds']}
    input_ = forward_kwargs.get('inputs_embeds')
    decoder_ = forward_kwargs.get('decoder_inputs_embeds')

    if decoder_ is None:
        forward_func = partial(model_forward, decoder_=decoder_, model=model, extra_forward_args=extra_forward_args)
        inputs = input_
    else:
        forward_func = partial(model_forward, model=model, extra_forward_args=extra_forward_args)
        inputs = (input_, decoder_)

    layer_ga = LayerGradientXActivation(forward_func, layer)
    attributions = layer_ga.attribute(inputs, target=prediction_id)[0]

    # if decoder_ is not None:
    #     # Does it make sense to concatenate encoder and decoder attributions before normalization?
    #     # We assume that the encoder/decoder embeddings are the same
    #     return normalize_attributes(torch.cat(attributions, dim=1))
    # else:
    return attributions #normalize_attributes(attributions)


def get_attr(ex_idx, args, feature_extractor, tokenizer, model, eval_df, template):
    image_extractor = CLIPFeatureExtractor.from_pretrained(args.encoder_name, cache_dir=CACHE_DIR)
    image_patches, image_id, caps, img = get_image_patches(eval_df, ex_idx, image_extractor, args)
    
    decoder_input_ids = prep_strings('', tokenizer, template=template, retrieved_caps=caps,
                                                k=int(args.k), is_test=True, max_length=140, reverse=args.reverse)
    decoder_input_len = len(decoder_input_ids)
    if args.features_path is not None:
        features = h5py.File(args.features_path, 'r')
    
    # get prediction
    encoder_last_hidden_state = torch.FloatTensor([features[image_id][()]])
    encoder_outputs = BaseModelOutput(last_hidden_state=encoder_last_hidden_state.to(args.device))
    
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#252b36'),
                                                  (1, '#000000')], N=256)
    
    with torch.no_grad():
        pred = model.generate(encoder_outputs=encoder_outputs,
                        decoder_input_ids=torch.tensor([decoder_input_ids]).to(args.device),
                        **args.generation_kwargs)
    
    input_tokens, pred_tokens = predid2tokens(pred, decoder_input_len, tokenizer)
    pixel_values = feature_extractor(img, return_tensors="pt").pixel_values
    
    
    attributions_all = []
    seq_len = len(pred.sequences[0])
    pred_len = len(pred_tokens)

    decoder_input_ids = torch.tensor([decoder_input_ids]).to(args.device)

    for idx in tqdm(range(seq_len-pred_len, seq_len)):
        prediction_id = pred.sequences[0][idx]    
        # decoder_input_embedding = interpretable_embedding.indices_to_embeddings(decoder_input_ids)
        decoder_input_embedding = model.decoder.transformer.wte(decoder_input_ids)
        attributions = compute_primary_attributions_scores(
            attr_method="ig",
            model=model,
            forward_kwargs={
                'inputs_embeds': pixel_values.requires_grad_().to(args.device),
                'decoder_inputs_embeds': decoder_input_embedding,
            },
            prediction_id=prediction_id
        )#.cpu().detach().numpy()
        attributions_all.append(attributions)
        
        if decoder_input_ids is not None:
            assert len(decoder_input_ids.size()) == 2 # will break otherwise
            decoder_input_ids = torch.cat(
                [decoder_input_ids, torch.tensor([[prediction_id]], device=decoder_input_ids.device)],
                dim=-1
            )
        else:
            input_ids = torch.cat(
                [input_ids, torch.tensor([[prediction_id]], device=input_ids.device)],
                dim=-1
            )

            # Recomputing Attention Mask
            if getattr(model, '_prepare_attention_mask_for_generation'):
                assert len(input_ids.size()) == 2 # will break otherwise
                attention_mask = model._prepare_attention_mask_for_generation(input_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)
                attention_mask = attention_mask.to(pixel_values.device)
    return attributions_all, pred_tokens, input_tokens, img, pixel_values, pred_len, seq_len
                

def main(args):
    dir_path = "/scratch/project/dd-23-80/code/RobCap" 
    sys.path.append(str(dir_path))  ## add path
    
    register_model_and_config()

    split="val"
    data = load_data_for_inference(os.path.join(dir_path, "data/dataset_coco.json"), args.captions_path, split)
    eval_df = pd.DataFrame(data[split])
    feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder_name, cache_dir=CACHE_DIR)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN

    args.generation_kwargs = {'max_new_tokens': CAPTION_LENGTH, 'no_repeat_ngram_size': 0, 'length_penalty': 0.,
                        'num_beams': BEAM_SIZE, 'early_stopping': True, 'eos_token_id': tokenizer.eos_token_id,
                        "output_attentions": True, "return_dict_in_generate": True, "output_scores":True}

    outfile_name = '{}_{}_preds%s.json'.format(split, "coco", "_attn_vis")


    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## load model
    checkpoint_path=args.checkpoint_path
    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)
    model = AutoModel.from_pretrained(checkpoint_path, cache_dir=CACHE_DIR)
    model.eval()
    model.to(args.device)

    args.features_path = os.path.join(dir_path, "features")
    # print("features_path: ", args.features_path)

    np.random.seed(1)
    
    template = open(os.path.join(dir_path, args.template_path)).read().strip() + ' '
    
    for ex_id in range(10):
        attributions_all, pred_tokens, input_tokens, img, pixel_values, pred_len, seq_len = get_attr(ex_id, args, feature_extractor, tokenizer, model, eval_df, template)
        dump_path = os.path.join("/scratch/project/dd-23-80/code/RobCap/vis_attr", 'attr_dict_%d.obj'% ex_id)
        file_dump = open(dump_path, 'w') 
        pickle.dump(
            {
            "attributions_all": attributions_all,
            "pred_tokens": pred_tokens,
            "input_tokens": input_tokens,
            "img": img,
            "pixel_values": pixel_values,
            "pred_len": pred_len,
            "seq_len": seq_len    
                }, file_dump)
        
        print("attrbution saved to %s" % dump_path)
    



if "__main__" in __name__:
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    PAD_TOKEN = '!'
    EOS_TOKEN = '.'
    CAPTION_LENGTH = 25
    BEAM_SIZE = 3

    CACHE_DIR = "/scratch/project/dd-23-80/cache"
    # CACHE_DIR = Path.home() / ".cache/huggingface/transformers/models" / CHECKPOINT_PATH

    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--images_dir", type=str, default="./data/images/", help="Directory where input image features are stored")
    parser.add_argument("--features_path", type=str, default=None, help="H5 file with cached input image features")
    parser.add_argument("--annotations_path", type=str, default="./data/dataset_coco.json", help="JSON file with annotations in Karpathy splits")
        
    parser.add_argument("--model_path", type=str, default=None, help="Path to model to use for inference")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to use for inference; If not specified, will infer with all checkpoints")

    parser.add_argument("--infer_test", action="store_true", default=False, help="Use test data instead of val data")

    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-base-patch32", help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="gpt2", help="Decoder name as found of HuggingFace or stored locally")

    parser.add_argument("--disable_rag", action="store_true", default=False, help="Disable retrieval augmentation or not")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved captions to use in prefix")
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64", help="Visual encoder used for retieving captions")
    parser.add_argument("--captions_path", type=str, default="./data/retrieved_caps_resnet50x64.json", help="JSON file with retrieved captions")
    parser.add_argument("--template_path", type=str, default="./src/template.txt", help="TXT file with template")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size; only matter if evaluating a norag model")

    # additional parameters for robust-cap
    parser.add_argument("--reverse", action="store_true",  default=False, help="if True reverse order of retrieved captions")
    parser.add_argument("--outfile_postfix", type=str, default='', required=False,
                        help="a customized postfix to be added to the inference output file")
    parser.add_argument("--dataset", type=str, default="coco", help="Use xm3600 data instead of coco data")


    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    main(args)