
# get the query 
import json
import os
import argparse
first_gen_files = {
    "elife":{
        "train":"data/GT_firstGen_elife.json",
        "test":"data/GT_firstGen_elife_test.json",
        "val":"data/GT_firstGen_elife_val.json"
    },
    "plos":{
        "train":"data/GT_firstGen_plos.json",
        "test":"data/GT_firstGen_plos_test.json",
        "val":"data/GT_firstGen_plos_val.json"
    }
}

plos_train_dir = "search_spaces/plos-wiki/search_space_train"
plos_val_dir = "search_spaces/plos-wiki/search_space_eval"

elife_train_dir = "search_spaces/elife-wiki/search_space_train"
elife_val_dir = "search_spaces/elife-wiki/search_space_eval"


# Change the data to the bge acceptable format 
plos_wiki_search_space = {
    "train":plos_train_dir,
    "val": plos_val_dir
}

elife_wiki_search_space = {
    "train":elife_train_dir,
    "val": elife_val_dir
}


def split_set(dataset, fold, ratio=0.95):
    datalen = len(dataset)
    if fold == "train":
        dataset = dataset[:int(ratio * datalen)]
    else:
        dataset = dataset[int(ratio * datalen):]
    return dataset

def generate_bge_file(args):
    if args.dataset=="elife":
        total_train_file = first_gen_files["elife"]["train"]
        train_list = os.listdir(elife_wiki_search_space["train"])
        val_list = os.listdir(elife_wiki_search_space["val"])
        train_dir = elife_wiki_search_space["train"]
        val_dir = elife_wiki_search_space["val"]
    elif args.dataset=="plos":
        total_train_file = first_gen_files["plos"]["train"]
        train_list = os.listdir(plos_wiki_search_space["train"])
        val_list = os.listdir(plos_wiki_search_space["val"])
        train_dir = plos_wiki_search_space["train"]
        val_dir = plos_wiki_search_space["val"]
    with open(total_train_file, "r") as f:
        data = json.load(f)
    
    train_queries = split_set(data,"train")
    val_queries = split_set(data,"val")
    
    dump_file = f"data/bge_{args.fold}_{args.knowledge}_{args.dataset}.jsonl"
    # print("len(train_list)", len(train_list))
    # print("len(train_queries)", len(train_queries))
    assert len(train_list) == len(train_queries)

    with open(dump_file, "w") as writein:
        if args.fold == "train":
            data_list = train_list
            data_query = train_queries
        else: 
            data_list = val_list
            data_query = val_queries
        for train, queries  in zip(data_list, data_query):
            item = {}
            with open(os.path.join(train_dir, train), "r") as f:
                train_split = json.load(f)
                negs = train_split["q"]
                poses = train_split["r"]
                new_pos = [doc["document"] for doc in poses]
                assert len(negs)==len(poses)
                neg_ids = []
                pos_ids = []
                for neg, pos in zip(negs, poses):
                    neg_ids.append(neg["docid"])
                    pos_ids.append(pos["docid"])
                new_neg_ids = []
                new_neg = []
                for i, j in zip(neg_ids, negs):
                    if i not in pos_ids:
                        new_neg_ids.append(i)
                        new_neg.append(j["document"])
                
                item["query"] = queries["first_gen"]
                item["pos"] = new_pos
                item["neg"] = new_neg
                
                json_str = json.dumps(item)
                json_str+="\n"
                writein.write(json_str)
                        
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="elife")
    parser.add_argument("--knowledge", type=str, default="wiki")
    parser.add_argument("--fold", type=str, default="train")

    args = parser.parse_args()  
    generate_bge_file(args)
