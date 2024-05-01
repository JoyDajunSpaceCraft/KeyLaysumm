# KeyLaysum 
This Repo is for the layman's summary and to achieve the task over the [Biolaysumm 2024](https://biolaysumm.org/) in 2024.

## Baseline model for LED model training

For training the LED model

PLOS: `python generator_train.py --dataset plos --output_dir laysumm_PLOS_cpt --output_file output/plos_test.txt`.

eLife: `python generator_train.py --dataset elife --output_dir laysumm_eLife_cpt --output_file output/elife_test.txt`.

The trained checkpoints can also find in this repo eLife [laysumm_eLife_cpt](laysumm_eLife_cpt) and [laysumm_PLOS_cpt](laysumm_PLOS_cpt).

For the first inference

```
python inference_rerank.py \
    --knowledge_base wiki \
    --generate_tokenizer allenai/led-base-16384 \
    --generate_model laysumm_PLOS_cpt \
    --device cude:0 \
    --n_docs 5 \
    --n_cands 8 \
    --max_input_length 16384 \
    --max_search_length 512 \
    --max_output_length 512 \
    --type first_inference \
    --input_file_path data/PLOS_val.jsonl \
    --output_file_path  output/plos.txt \
    --mix_knowledge_num 5 
```


## Create search space

Search space is for both train and evaluation. Run `python neural_reranker_colbert.py` to generate the search space. 
For every search space, every file represents one `ground truth` (key as q) and one for the `first generation` (key as r). 

Search space with the format for colbert reranker training:

```
{
    "q": [
        {
            "docid": "366112_11",
            "document": "Chikungunya . Currently, the cause of these chronic symptoms is not fully known. ...",
            "score": 0.0
        },
        ...
        ],
     "r": [
        {
            "docid": "54035498_2",
            "document": "Matthew Boulton (epidemiologist) ...",
            "score": 17.493637084960938
        },
        ...
        ]
}

```

This format follows the [KARD](https://github.com/Nardien/KARD) reranker.

for the neural reranked training for different knowledge 
Now we support Wikipedia and bioasq.
For different training rerankers, now we are using [Colbert](https://arxiv.org/pdf/2004.12832.pdf) and [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) for the score calculation.

### Change the search space to support bge reranker 

Running the code in the format that fits into the format of the BAAI/bge-reranker-v2-m3 training JSONL.

`python generate_bge_file.py --knowledge wiki --dataset elife --fold train`

`python generate_bge_file.py --knowledge wiki --dataset plos --fold train`

Then the data should be `data/bge_train_wiki_elife.jsonl` and `data/bge_train_wiki_plos.jsonl`.


## Train Colbert and Bge
If multiple devices

For colbert: `CUDA_VISIBLE_DEVICES=0 python neural_reranker_colbert.py --dataset plos  --save_dir colbert_lr1e-3_plos_train --file_name GT_firstGen_plos_train.json` and 
`CUDA_VISIBLE_DEVICES=0 python neural_reranker_colbert.py --dataset elife  --save_dir colbert_lr1e-3_elife_train --file_name GT_firstGen_elife_train.json`.

For bge: `CUDA_VISIBLE_DEVICES=0 python neural_reranker_baai_bge.py  --data_type plos` and  `CUDA_VISIBLE_DEVICES=0 python neural_reranker_baai_bge.py  --data_type elife`. 


## Generate the retrieval augmented generation 

For the retrieval augmented generation with the colbert

```
python inference_rerank.py \
    --knowledge_base wiki \
    --retriever_model colbert \
    --search_tokenizer save/colbert_lr1e-3 \
    --generate_tokenizer allenai/led-base-16384 \
    --dense_retriever_path save/colbert_lr1e-3/model \
    --generate_model laysumm_PLOS_cpt \
    --n_docs 5 \
    --n_cands 8 \
    --max_input_length 16384 \
    --max_search_length 512 \
    --max_output_length 512 \
    --type rag_inference \
    --input_file_path output/plos.txt \
    --output_file_path output/plos_second_colbert.txt \
    --mix_knowledge_num 5 
```

For the retrieval augmented generation with the bge

```
python inference_rerank.py \
    --knowledge_base wiki \
    --retriever_model bge \
    --dense_retriever_path save/bge_models_plos \
    --search_tokenizer save/bge_models_plos \
    --generate_tokenizer allenai/led-base-16384 \
    --generate_model laysumm_PLOS_cpt \
    --n_docs 5 \
    --n_cands 8 \
    --max_input_length 16384 \
    --max_search_length 512 \
    --max_output_length 512 \
    --type rag_inference \
    --input_file_path output/plos.txt \
    --output_file_path output/plos_second_bge.txt \
    --mix_knowledge_num 5 
```

## Generate ChatGPT reasoning generation 

Run the chatgpt results here `chatgpt_inference.ipynb`

## Control generation

1. We use the PPO as the function for the model generation 

Implemantation detail:

`python control_generation.py`


## Evaluation 

For evaluation, please check [Evaluation 2024 share task](https://github.com/TGoldsack1/BioLaySumm2024-evaluation_scripts).