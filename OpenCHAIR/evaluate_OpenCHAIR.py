import pandas as pd
from tqdm.contrib import tzip
from tqdm import tqdm

import torch
import argparse

from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset

from utils import (
    extract_objs, 
    load_llm_pipe, 
    get_answer
)


def generate_captions(och_dataset, args):
    # load model
    processor = BlipProcessor.from_pretrained(args.model_ckpt, cache_dir=args.cache_dir)
    model = BlipForConditionalGeneration.from_pretrained(
        args.model_ckpt, torch_dtype=torch.float16, cache_dir=args.cache_dir
    )
    model.to(args.device)

    generated_captions = []

    # Generate Captions
    with tqdm(total=len(och_dataset)) as pbar:
        for data in tqdm(och_dataset.iter(batch_size=args.batch_size)):
            inputs = processor(text=[args.prompt for _ in range(len(data['image']))] , images=data['image'], return_tensors="pt", 
                               padding=True, truncation=True).to(args.device, torch.float16)

            with torch.inference_mode():
                generated_ids = model.generate(**inputs,
                                               num_beams=args.beam_size)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_texts = [text[len(args.prompt):].strip() for text in generated_texts]
            generated_captions += generated_texts
            pbar.update(args.batch_size)
    return generated_captions

def get_llm_responses(args, och_captions, generated_objs):
    llm_pipe = load_llm_pipe(args)
    hits = []
    for cap, objs in tzip(och_captions, generated_objs):
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
    print("Loading Dataset\n")
    och_dataset = load_dataset("moranyanuka/OpenCHAIR", cache_dir=args.cache_dir)['test']
    print("\nGenerating Captions\n")
    generated_captions = generate_captions(och_dataset, args)
    och_captions = och_dataset['text']

    word_conc = pd.read_excel(args.concreteness_dataset_path)[['Word','Conc.M']].set_index("Word").to_dict()['Conc.M']
    print("\nExtracting Generated Object\n")
    generated_objs = extract_objs(generated_captions, word_conc)

    print("\nGetting LLM Responses\n")
    llm_responces = get_llm_responses(args, och_captions, generated_objs)
    OpenCHAIR_score = get_och_score(llm_responces)
    print("\nOpenCHAIR Score: \n")
    print(OpenCHAIR_score)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", type=str, default="moranyanuka/blip-image-captioning-base-mocha")
    parser.add_argument("--llm-ckpt", type=str, default="meta-llama/Llama-2-70b-chat-hf")
    parser.add_argument("--concreteness-dataset-path", type=str, default="./OpenCHAIR/Concreteness_ratings_Brysbaert_et_al_BRM.xlsx")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()
    eval(args)
    
    