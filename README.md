# Robust-cap
retrieval robust captioning 

## Dependencies

The code is based on was developed in Python 3.9. and is based on [SmallCap](https://github.com/RitaRamo/smallcap).

```
conda create -n robcap python=3.9
conda activate robcap
pip install -r requirements.txt
```

## Preparation
Follow the data downloading and preprocessing process as in [SmallCap](https://github.com/RitaRamo/smallcap).
A preprocessed larger list of retrieved captions can be downloaded [here](https://drive.google.com/file/d/17OdjGGTr-6dDPhSvQ1IQM8QOa3_Ry42K/view?usp=sharing). You can also follow the [instructions](https://github.com/RitaRamo/smallcap) to retrieve captions by yourself.

### Preprocess
```
python src/preprocess/permute_retrieved_caps.py --input_file <input_file_path> --method permute --topk 4 
```

#### Evaluation package
Use this version of pycocoeval, SPICE model would be automatically installed, and it is possible to calculate CLIPScore (Recommended)  
`pip install git+https://github.com/jmhessel/pycocoevalcap`

Otherwise, you can download Stanford models for computing SPICE (a slightly modified version of this [repo](https://github.com/daqingliu/coco-caption.git)):

```./coco-caption/get_stanford_models.sh```

After the `pycocoevalcap` is installed you can run:
`python src/run_eval.py <GOLD_ANN_PATH> <PREDICTIONS_PATH>`  
output results are saved in the same folder as the  `<PREDICTIONS_PATH>`
</details>

e.g. ```python src/run_eval.py coco-caption/annotations/captions_valKarpathy.json baseline/rag_7M_gpt2/checkpoint-88560/val_preds_original.json```


## Visualization
- `src/vis` contains two notebook which visualizes the attention plots (`vis_attn.ipynb`) for decoder self attention and cross attention (`vis_cross_attn.ipynb`).

- `get_attn_layer_distr.py` and `get_prompt_token_attn_distr.py` are helper scripts that extract maximum attention scores at a layerwise or a tokenwise manner for visualization use. (these might need to be organized later)


## Train
```
export ORDER="sample"
python train.py \
  --experiments_dir $EXP \
  --captions_path $CAPTIONS_PATH \
  --decoder_name facebook/opt-350m \
  --attention_size 1.75 \
  --batch_size 64 \
  --n_epochs 10 \
  --order $ORDER \
  --k 4 
```

## Eval
```
export ORDER="default"
python infer.py --model_path $MODEL_PATH --checkpoint_path checkpoint-88560 \
  --decoder_name "facebook/opt-350m" \
  --captions_path $CAPTIONS_PATH \
  --order $ORDER \
  --outfile_postfix _$ORDER
```

and calculate scores with:

```
python src/run_eval.py \
    coco-caption/annotations/captions_valKarpathy.json \
   $MODEL_PATH/checkpoint-88560/val_coco_preds_$ORDER.json
```


## Citation
```
@article{li2024understanding,
  title={Understanding Retrieval Robustness for Retrieval-Augmented Image Captioning},
  author={Li, Wenyan and Li, Jiaang and Ramos, Rita and Tang, Raphael and Elliott, Desmond},
  journal={arXiv preprint arXiv:2406.02265},
  year={2024}
}
```












