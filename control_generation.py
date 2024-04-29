from typing import List
from readability import Readability
from datasets import load_dataset
import numpy as np
import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

try:
    import evaluate
except ImportError:
    raise ImportError(
        "To run this example, please install the `evaluate` and `nltk` packages" "by running `pip install evaluate`"
    )

config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=1,# default 100
        total_steps=100000,
        batch_size=12,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
    ),
    model=ModelConfig(
        model_path="google/flan-t5-large",
        model_arch_type="seq2seq",
        num_layers_unfrozen=2,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="google/flan-t5-large",
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1.0e-5,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 10000,
            "eta_min": 1.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=512,
        chunk_size=12,
        ppo_epochs=4,# default 4
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=0.99,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1.0,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 1024,
        },
        gen_experience_kwargs={
            "max_new_tokens": 1024,
            # "min_new_tokens":1024,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
        },
    ),
)



import textstat

def get_flesch(text):
    return textstat.flesch_kincaid_grade(text)
def get_dcrs(text):

    return textstat.dale_chall_readability_score(text)
def get_cli(text):
    return textstat.coleman_liau_index(text)



sigma = 10
def calc_nd(value, mean):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (value - mean) ** 2 / (2 * sigma ** 2)) / 0.039894228040143274


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import os
model_name = "google/flan-t5-large"
device = "cuda:" + str(os.environ.get('LOCAL_RANK',0))
num_layers = 17
cache_dir=".cache"
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model.encoder.block[0].layer = torch.nn.ModuleList([layer for layer in model.encoder.block[0].layer[:num_layers]])
from bert_score import score

import numpy as np
def calc_bertscore(preds, refs):
  # Get BERTScore F1 scores
  P, R, F1 = score(preds, refs, lang="en", verbose=True, device='cuda:0')
  return np.mean(F1.tolist())

def calc_mean(scores):
    return np.mean(scores)


wandb.init(project='trlx_summarization', entity='PLOS_val')

def reward_fn(samples: List[str], prompts: List[str], outputs: List[str], tokenizer= tokenizer):
    flesch_scores = []
    original_scores = []
    summaries = []
    docs = []
    for (generated_summary, input_doc) in zip(outputs, prompts):
        docs.append(input_doc)
        summaries.append(generated_summary.strip())
        try:
            temp = []
            temp.append(get_flesch(generated_summary.strip()))
            temp.append(get_dcrs(generated_summary.strip()))
            temp.append(get_cli(generated_summary.strip()))

            print("generated scores", temp)
            flesch_scores.append(sum(temp)/3)
        except:
            flesch_scores.append(0)

    all_bertscore_scores = []
    for doc, summary in zip(docs, summaries):
        bertscore_scores = calc_bertscore([doc], [summary])
        all_bertscore_scores.append(bertscore_scores)

    # assert len(original_scores) == len(flesch_scores) == len(all_bertscore_scores)
    assert len(flesch_scores) == len(all_bertscore_scores)
    mean = calc_mean(flesch_scores)
    # print("flesch_scores1", flesch_scores)

    # flesch_scores = [calc_nd(fs, o_fs) for fs, o_fs in zip(flesch_scores, original_scores)]
    flesch_scores = [calc_nd(fs, mean) for fs in flesch_scores]


    # New generation

    length_scores = [len(output.strip().split()) for output in outputs]  # the length represent the words size

    # Normalized the length
    max_length = max(length_scores)
    length_scores = [score / max_length for score in length_scores]  # Normalized make them between 0-1

    # Change to tensor
    flesch_scores = torch.tensor(flesch_scores, dtype=torch.float32)
    all_bertscore_scores = torch.tensor(all_bertscore_scores, dtype=torch.float32)
    length_scores = torch.tensor(length_scores, dtype=torch.float32)

    # Normalized Flesch make it the larger the better
    max_flesch_score = max(flesch_scores)
    min_flesch_score = min(flesch_scores)
    normalized_flesch_scores = (max_flesch_score - flesch_scores) / (max_flesch_score - min_flesch_score)

    # Combine three scores: normalized_flesch_scores, BERTScore (all_bertscore_scores), and length (length_scores)
    # You can adjust the following weights to control the importance of each reward
    readability_weight = 0.5
    bertscore_weight = 0.3
    length_weight = 0.2

    # Calculate the overall score 
    combined_scores = readability_weight * normalized_flesch_scores + \
                      bertscore_weight * all_bertscore_scores + \
                      length_weight * length_scores

    # Convert tensor to list and return
    final_scores = combined_scores.tolist()
    wandb.log({"Average Flesch Score": np.mean(flesch_scores),
               "Average BERT Score": np.mean(bert_scores),
               "Average Final Score": np.mean(final_scores)})
    print("final_scores", final_scores)
    return final_scores

# dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir="data")
train_data_file = "data/GT_firstGen_elife.json"
eval_data_file = "data/GT_firstGen_elife_test.json"
raw_datasets = load_dataset("json", data_files={'train': train_data_file, "eval":eval_data_file})

# take 20,000 samples from the training set as prompts for training
prompts = raw_datasets["train"]["first_gen"]
summaries = raw_datasets["train"]["ground_truth"]


# take 1,000 samples from the validation set as prompts for evaluation
val_prompts = raw_datasets["eval"]["first_gen"]
val_summaries = raw_datasets["eval"]["ground_truth"]

model = trlx.train(
    reward_fn=reward_fn,
    prompts=[{"prompt": prompt} for prompt in prompts],
    eval_prompts=[{"prompt": prompt} for prompt in val_prompts],
    config=config,
).model
# Please add your wandb number 


# Evaluation 
from transformers import T5ForConditionalGeneration, T5Tokenizer
model = T5ForConditionalGeneration.from_pretrained("ckpts/best_checkpoint/hf_model")
tokenizer = T5Tokenizer.from_pretrained("ckpts/best_checkpoint/hf_model")
# 
output = model.generate(**tokenizer(["An astronaut riding "] * 16, return_tensors="pt"))
tokenizer.batch_decode(output, skip_special_tokens=True)
input_text = "translate English to French: How are you?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

outputs = model.generate(input_ids)

print(tokenizer.decode(outputs[2], skip_special_tokens=True))

