import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["NCCL_DEBUG"] = "INFO"
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import evaluate

CKPT_BASE = "allenai/led-base-16384"
MAX_INPUT_LENGTH = 16384
MAX_OUTPUT_LENGTH = 512
BATCH_SIZE = 1
EPOCHS = 1

files = {
    "elife":{
        "train":"data/eLife_train.jsonl",
        "val": "data/eLife_val.jsonl",
        "test":"data/eLife_test.jsonl"
    },
    "plos":{
        "train":"data/PLOS_train.jsonl",
        "val": "data/PLOS_val.jsonl",
        "test":"data/PLOS_test.jsonl"
    }
}
def train(args):
    tokenizer = AutoTokenizer.from_pretrained(CKPT_BASE)
    
    train_data_file = files[args.dataset]["train"]
    eval_data_file = files[args.dataset]["val"] 

    test_datat_file = files[args.dataset]["test"] 
    raw_datasets = load_dataset("json", data_files={'train': train_data_file, "eval":eval_data_file, "test":eval_data_file})

    def process_data_to_model_inputs(batch):
        inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=MAX_INPUT_LENGTH)
        outputs = tokenizer(batch["lay_summary"], padding="max_length", truncation=True, max_length=MAX_OUTPUT_LENGTH)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = outputs.input_ids
        batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

        return batch

    tokenized_datasets = raw_datasets.map(process_data_to_model_inputs, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT_BASE)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=EPOCHS,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        
        train_dataset=tokenized_datasets["train"],
        # eval_dataset=tokenized_datasets["eval"],
        tokenizer=tokenizer,
    )


    trainer.train()
    output_merged_dir = args.output_dir
    model.save_pretrained(output_merged_dir, safe_serialization=False)

    # trainer.evaluate()

    test_dataset = tokenized_datasets["test"]
    predictions = trainer.predict(test_dataset)
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)


    output_file = args.output_file
    with open(output_file, 'w') as f:
        for summary in decoded_preds:
            f.write(summary + '\n')

    print(f"Generated summaries are saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="plos")
    parser.add_argument("--output_dir", type=str, default="laysumm_PLOS_cpt")
    parser.add_argument("--output_file",type=str, default="plos_test.txt" )
    args = parser.parse_args()
    train(args)