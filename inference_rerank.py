import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PretrainedConfig, AutoModelForSeq2SeqLM,AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from einops import rearrange
from typing import Dict
from termcolor import cprint
import argparse
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher




class Retriever:
    def __init__(self, args):

        self.query_model_path = args.dense_retriever_path
        self.retriever_model = args.retriever_model

        self.device = args.device
        if self.query_model_path is None or args.retriever_model == "sparse":
            self.query_model, self.key_model, self.tokenizer = None, None, None
            self.max_seq_len = 512
        elif self.retriever_model == "colbert":
            print("self.query_model_path", self.query_model_path)
            _config = PeftConfig.from_pretrained(self.query_model_path)
            print("_config",_config)
            base_model = _config.base_model_name_or_path
            print("base_model", base_model)
            self.tokenizer = AutoTokenizer.from_pretrained(args.search_tokenizer)

            self.query_model = ColBERT.from_pretrained(base_model)
            self.query_model = PeftModel.from_pretrained(self.query_model, self.query_model_path)
            self.query_model.eval()
            self.query_model.to(self.device)
            self.max_seq_len = min(self.tokenizer.model_max_length, 512)
        elif self.retriever_model=="bge":
            self.tokenizer = AutoTokenizer.from_pretrained(args.search_tokenizer)
            self.query_model = AutoModelForSequenceClassification.from_pretrained(self.query_model_path)
            self.query_model.eval()
            self.query_model.to(self.device)
        else:
            raise NotImplementedError


    def retrieve(self, q, documents, n=1, n_cands=-1, return_rank=False):
        if self.query_model is None: return documents[0]

        if self.retriever_model == "bge":
            pairs = [[q, doc] for doc in documents]
            with torch.no_grad():
                inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
                scores = self.query_model(**inputs, return_dict=True).logits.view(-1, ).float()
            ranking = sorted([(idx, score) for idx, score in enumerate(scores)], key=lambda x: x[1], reverse=True)
            documents = sorted([(doc, score) for doc, score in zip(documents, scores)], key=lambda x: x[1], reverse=True)
            documents = [doc[0] for doc in documents]
            
        elif self.retriever_model in ["colbert"]:
            query_outputs = self.tokenizer(q, return_tensors='pt', max_length=self.max_seq_len, padding='max_length', truncation=True)
            key_outputs = self.tokenizer(documents, return_tensors='pt', max_length=self.max_seq_len, padding='max_length', truncation=True)

            query_inputs = {k:v.to(self.device) for k, v in query_outputs.items()}
            key_inputs = {k:v.to(self.device) for k, v in key_outputs.items()}
            # scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            with torch.no_grad():
                similarity = self.query_model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = similarity.flatten().tolist()
            ranking = sorted([(idx, score) for idx, score in enumerate(scores)], key=lambda x: x[1], reverse=True)
            documents = sorted([(doc, score) for doc, score in zip(documents, scores)], key=lambda x: x[1], reverse=True)
            documents = [doc[0] for doc in documents]

        else:
            raise NotImplementedError

        if return_rank: return documents, ranking
        return documents

class ColBERTConfig(PretrainedConfig):
    compression_dim: int = 768
    dropout: float = 0.0
    return_vecs: bool = False
    trainable: bool = True

class ColBERT(PreTrainedModel):
    """
    ColBERT model from: https://arxiv.org/pdf/2004.12832.pdf
    We use a dot-product instead of cosine per term (slightly better)
    """
    config_class = ColBERTConfig
    base_model_prefix = "bert_model"

    def __init__(self, cfg, n_cands=8) -> None:
        super().__init__(cfg)
        
        self.bert = AutoModel.from_pretrained(cfg._name_or_path)

        # for p in self.bert.parameters():
        #     p.requires_grad = cfg.trainable

        self.compressor = torch.nn.Linear(self.bert.config.hidden_size, cfg.compression_dim)

        self.n_cands = n_cands
        print(f"Model n_cands: {self.n_cands}")

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor]):

        query_vecs = self.forward_representation(query)
        document_vecs = self.forward_representation(document)

        score = self.forward_aggregation(query_vecs, document_vecs, query["attention_mask"], document["attention_mask"])
        return score

    def forward_representation(self,
                               tokens,
                               sequence_type=None) -> torch.Tensor:
        
        vecs = self.bert(**tokens)[0] # assuming a distilbert model here
        vecs = self.compressor(vecs)

        # # if encoding only, zero-out the mask values so we can compress storage
        # if sequence_type == "doc_encode" or sequence_type == "query_encode": 
        #     vecs = vecs * tokens["tokens"]["mask"].unsqueeze(-1)

        return vecs

    def forward_aggregation(self, query_vecs, document_vecs, query_mask, document_mask):
        # query_vecs: B x N x D
        # doc_vecs: (B * k) x N x D

        # Unsqueeze query vector
        _bsz = query_vecs.shape[0]
        n_cands = document_vecs.shape[0] // _bsz
        query_vecs_dup = query_vecs.repeat_interleave(n_cands, dim=0).contiguous()

        score = torch.bmm(query_vecs_dup, document_vecs.transpose(1, 2))
        exp_mask = document_mask.bool().unsqueeze(1).expand(-1, score.shape[1], -1)
        score[~exp_mask] = - 10000

        # max pooling over document dimension
        score = score.max(-1).values
        query_mask_dup = query_mask.repeat_interleave(n_cands, dim=0).contiguous()

        score[~(query_mask_dup.bool())] = 0
        score = rearrange(score.sum(-1), '(b n) -> b n', n=n_cands) # B x k 
        return score



def inference(args):
    
    tokenizer = AutoTokenizer.from_pretrained(args.generate_tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.generate_model)
    model.to(args.device)
    model.eval()
    if args.knowledge_base=="wiki":
        knowledge_name="enwiki-paragraphs"
    elif args.knowledge_base=="pubmed":
        knowledge_name="beir-v1.0.0-bioasq-flat"
    

    with open(args.input_file_path, 'r') as f, open(args.output_file_path, "w") as output_file:
        if args.type=="first_inference":
            data = []
            for i in f.readlines():
                read_file = json.loads(i)
                # data.append(read_file["article"])  
                input_ids = tokenizer.encode(read_file["article"], return_tensors="pt")
                input_ids = input_ids.to(args.device) 
           
                with torch.no_grad():  
                    generated_ids = model.generate(
                        input_ids,
                        max_length=args.max_input_length + args.max_output_length,
                        num_beams=5,
                        no_repeat_ngram_size=2,
                        early_stopping=True
                    )
    
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print("Generated text:", generated_text)
    
                output_file.write(generated_text + "\n") 
                
        elif args.type=="rag_inference":
            data = []
            for i in f.readlines():
                data.append(i) 
            dense_retriever = Retriever(args)

            for line in tqdm(data):
            
                search_q = line.split("\n")[0]
                if len(search_q.split(" "))>1024:
                    search_q = search_q[:1024]
                
                searcher = LuceneSearcher.from_prebuilt_index(knowledge_name)
                docs = []
                for hit in searcher.search(search_q,k=args.mix_knowledge_num):
                    res = searcher.doc(hit.docid).raw()
                    docs.append(res)
                documents = dense_retriever.retrieve(search_q, docs, n=args.n_docs, n_cands=args.n_cands)
                knowledge = documents[:args.n_docs]
                new_input = search_q + " ".join(knowledge)
                # need to know the max input search size
                if len(new_input) > args.max_search_length:
                    new_input = new_input[:args.max_search_length]
    
                input_ids = tokenizer.encode(new_input, return_tensors="pt")
                input_ids = input_ids.to(args.device) 
    
           
                with torch.no_grad():  
                    generated_ids = model.generate(
                        input_ids,
                        max_length=args.max_input_length + args.max_output_length,
                        num_beams=5,
                        no_repeat_ngram_size=2,
                        early_stopping=True
                    )
    
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print("Generated text:", generated_text)
    
                output_file.write(generated_text + "\n")



if __name__ == "__main__":
    model_path = "save/colbert_lr1e-3/model/"
    generate_model_path = "laysumm_PLOS_cpt"
    tokenized_path = "allenai/led-base-16384"
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge_base", type=str, default="wiki")
    parser.add_argument("--retriever_model", type=str, default="colbert")
    parser.add_argument("--dense_retriever_path", type=str, default=model_path)
    parser.add_argument("--search_tokenizer",type=str, default=model_path)
    parser.add_argument("--generate_tokenizer",type=str, default=tokenized_path)
    parser.add_argument("--generate_model", type=str, default=generate_model_path)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_docs", type=int, default=1)
    parser.add_argument("--n_cands", type=int, default=8)
    
    parser.add_argument("--max_input_length", type=int, default=16384)
    parser.add_argument("--max_search_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=512)
    parser.add_argument("--output_file_path", type=str, default= "plos_second.txt")
    parser.add_argument("--input_file_path", type=str, default="plos.txt")
    parser.add_argument("--type", type=str, default="first_inference")
    parser.add_argument("--mix_knowledge_num", type=int, default="5")

    args = parser.parse_args()  
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    inference(args)
