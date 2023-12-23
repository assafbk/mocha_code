import argparse
import torch
from tqdm import tqdm
import pandas as pd

from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset

def generate(och_dataset, args):
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
            inputs = processor(text=[args.prompt for _ in range(len(data['image']))],
                               images=data['image'], return_tensors="pt", 
                               padding=True, truncation=True).to(args.device, torch.float16)

            with torch.inference_mode():
                generated_ids = model.generate(**inputs,
                                               num_beams=args.num_beams)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_texts = [text[len(args.prompt):].strip() for text in generated_texts]
            generated_captions += generated_texts
            pbar.update(args.batch_size)
    return generated_captions

def run(args):
    print("Loading Dataset\n")
    och_dataset = load_dataset("moranyanuka/OpenCHAIR", cache_dir=args.cache_dir)['test']
    print("\nGenerating Captions\n")
    generated_captions = generate(och_dataset, args)

    df = pd.DataFrame()
    df['generated_caption'] = generated_captions
    df.to_csv(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt", type=str, default="moranyanuka/blip-image-captioning-base-mocha")
    parser.add_argument("--prompt", type=str, default="a photography of ")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./OpenCHAIR/out.csv")
    args = parser.parse_args()
    run(args)