from datasets import Dataset, DatasetDict, load_dataset,DatasetBuilder
import pandas as pd



def main():
    df_dict = {
        "train": pd.read_pickle("df_annotated_train.picle"),
        "test": pd.read_pickle("df_annotated_test.picle"),
        "valid": pd.read_pickle("df_annotated_val.picle"),
    }
    for key in df_dict:
        df = df_dict[key]
        df["tokens"]=df["new_tokens"]
        df["ner_tags"]=df["new_ner_tags"]
        df_dict[key] = df[["id","tokens","ner_tags"]]
        
    for key in df_dict:
        df = df_dict[key] 
        df.to_json(f"../data/{key}.json", orient="records",lines=True)


if __name__ == '__main__':
    main()