import torch
import json

from tqdm import tqdm
from openai import OpenAI

import openai


predictions = []
dense_retriever = Retriever(args)
for line in tqdm(first_gen):
    # item = json.loads(line)
    # article = item["article"]
    search_q = line.split("\n")[0]
    searcher = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')
    # searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-bioasq-flat')
    docs = []
    for hit in searcher.search(search_q):
        res = searcher.doc(hit.docid).raw()
        docs.append(res)
    documents = dense_retriever.retrieve(search_q, docs, n=args.n_docs, n_cands=args.n_cands)
    knowledge = documents[:args.n_docs]
    knowledge = " ".join(knowledge)
    # print("knowledge", knowledge)
    response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system," "content": "You are a layman rephrase; your goal is to rephrase the input and make it easier to read.\
        For example 'Diabetes is a condition in which the pancreas cannot produce enough insulin to feed the body. \
        This is caused by a protein called proinsulin is an ingredient which is made up of a group of molecules called cysteine thiols.'\
        The rephrased result should be: 'Diabetes is a condition where the pancreas doesn't produce enough insulin to meet the body's needs. \
        This happens because of a protein called proinsulin, which consists of a group of molecules known as cysteine thiols.'
         "},
        {"role": "user", "content": f'Here is the original text I want you to help me to rephrase: {line}.\
        Make it easier to read and retain as much the original text and have a similar length as the original text. '},
      ],
        
    )
    res = response.choices[0].message.content
    predictions.append(res)
   