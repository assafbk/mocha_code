import pandas as pd
from tqdm.contrib import tzip
import argparse

from utils import (
    extract_objs, 
    load_llm_pipe, 
    get_answer
)

def get_llm_responses(args, df):
    llm_pipe = load_llm_pipe(args)
    hits = []
    for cap, objs in tzip(df.gt_caption, df.generated_objs):
        objs = set(objs)
        for obj in objs:
            hits.append(get_answer(cap, obj, llm_pipe))
    return hits

def get_och_score(responses):
    data = pd.Series(responses).str.lower().str.strip()
    dv = data.value_counts()
    d = dv.to_dict()
    return 1 - d['yes'] / dv.sum()


def eval(args):
    df = pd.read_csv(args.generations_file_path)

    word_conc = pd.read_excel(args.concreteness_dataset_path)[['Word','Conc.M']].set_index("Word").to_dict()['Conc.M']
    print("\nExtracting Generated Object\n")
    df['generated_objs'] = extract_objs(df.generated_caption.tolist(), word_conc)

    print("\nGetting LLM Responses\n")
    llm_responces = get_llm_responses(args, df)
    OpenCHAIR_score = get_och_score(llm_responces)
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