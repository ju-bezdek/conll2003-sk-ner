from datasets import load_dataset
import argparse
from googleapiclient.discovery import build
import pandas as pd

def main(api_key:str, target_lang="sk", batch_size = 100):
    """
    Downloads conll2003 dataset and translates it to sk

    -api_key - google translate api key
    
    -target_lang - target language... default = sk
    
    -batch_size - > how many records would be in single google api call ... default = 100
    """

    conll2003 = load_dataset("conll2003")
    conll2003["train"]
    src_lang="en"   

    service = build('translate', 'v2', developerKey=api_key)

    pandas_dict={}
    for subset in conll2003.keys():
        pandas_dict[subset] = conll2003[subset].to_pandas()[["id","tokens","ner_tags"]]

    for subset in conll2003.keys():
        char_count=0
        l = len(pandas_dict[subset])
        all_outputs=[]
        for i in range(0,l,batch_size):
            print(i)
            inputs = list(pandas_dict[subset]["tokens"][i:min(i +batch_size, l)].map(lambda token_list: " ".join(token_list)))
            outputs = translate(service, inputs, src_lang, target_lang)
            all_outputs+=outputs
            #sentence = " ".join(pandas_dict[subset]["tokens"][i])
            #(" ".join(pandas_dict[subset][i]["tokens"]))
        print(len(all_outputs))
        pandas_dict[subset]["translated"]= all_outputs
        pandas_dict[subset].to_parquet(f"./translated/{subset}.parquet")


def translate(service, src_texts, src_lang, target_lang, return_dict=False):
    outputs = service.translations().list(source='en', target='sk', q=src_texts).execute()
    if return_dict:
        return {input: output['translatedText'] for (input, output) in zip(src_texts, outputs['translations'])}
    else:
        return [output['translatedText'] for output in outputs['translations']]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Download conll2003 dataset and translate it to target language')
    parser.add_argument('api_key', help='google translate api key')
    parser.add_argument('--target_lang', help='ttarget language... default = sk', default="sk")
    parser.add_argument('--batch_size', help='how many records would be in single google api call ... default = 100', default=100)
    args = parser.parse_args()
    
    main(**vars(args))