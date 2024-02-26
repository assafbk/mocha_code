import pandas as pd
from tqdm.contrib import tzip
import argparse
from datasets import load_dataset

from utils import (
    extract_objs, 
    load_llm_pipe, 
    get_answer
)

def get_llm_responses(df, llm_pipe):
    ignore_words = ['painting', 'drawing', 'photo', 'picture', 'portrait', 'photograph']
    hits = []
    for cap, objs in tzip(df.gt_caption, df.generated_objs):
        cur_hits = []
        for obj in objs:
            if obj in ignore_words:
                cur_hits.append('ignore')
            else:
                cur_hits.append(get_answer(cap, obj, llm_pipe))
        hits.append(cur_hits)
    return hits

def get_och_score(llm_responses):
    responses = []
    [responses.extend(resp_per_cap) for resp_per_cap in llm_responses]
    data = pd.Series(responses).str.lower().str.strip()
    dv = data.value_counts()
    d = dv.to_dict()
    return d['no'] / (d['yes'] + d['no'])


def eval(args):
    print("Loading Dataset\n")
    och_dataset = load_dataset("moranyanuka/OpenCHAIR", cache_dir=args.cache_dir)['test']
    df = pd.read_csv(args.generations_file_path)
    df['gt_caption'] = och_dataset['text']

    word_conc = pd.read_excel(args.concreteness_dataset_path)[['Word','Conc.M']].set_index("Word").to_dict()['Conc.M']
    print("\nExtracting Generated Object\n")
    df['generated_objs'] = extract_objs(df.generated_caption.tolist(), word_conc)

    print("\nLoading LLM\n")
    llm_pipe = load_llm_pipe(args)

    print("\nGetting LLM Responses\n")
    llm_responses = get_llm_responses(df, llm_pipe)
    OpenCHAIR_score = get_och_score(llm_responses)
    print("\nOpenCHAIR Score: \n")
    print(OpenCHAIR_score)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-ckpt", type=str, default="meta-llama/Llama-2-70b-chat-hf")
    parser.add_argument("--concreteness-dataset-path", type=str, default="./OpenCHAIR/Concreteness_ratings_Brysbaert_et_al_BRM.xlsx")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--generations-file-path", type=str, default="./OpenCHAIR/out.csv")
    args = parser.parse_args()
    eval(args)