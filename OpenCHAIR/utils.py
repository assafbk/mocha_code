
from tqdm.auto import tqdm
from itertools import islice
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
from functools import lru_cache
import spacy
import en_core_web_sm


def is_concrete(noun, concretness, t=4.5):
    if noun in concretness:
        return concretness[noun] > t
    return False

def extract_objs(captions, conc_df):
    nlp = en_core_web_sm.load()
    objs = []
    for caption in tqdm(captions):
        doc = nlp(caption.lower())
        cur_objs = [token.lemma_ for token in doc if token.pos_ == 'NOUN' and is_concrete(token.lemma_, conc_df)]
        objs.append(cur_objs)
    return objs

def load_llm_pipe(args):
    tokenizer = AutoTokenizer.from_pretrained(args.llm_ckpt)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_use_double_quant=True)
    
    model = AutoModelForCausalLM.from_pretrained(args.llm_ckpt,
                                                 quantization_config=bnb_config,device_map="auto",
                                                 cache_dir=args.cache_dir)
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    trust_remote_code=True,
                    device_map="auto")
    return pipe

def parse_ans(ans):
    ans_word_list = ans.lower().replace(',','').replace('.','').replace(';','').replace('\n',' ').split(' ')
    if 'yes' in ans_word_list:
        return 'yes'
    elif 'no' in ans_word_list or 'not' in ans_word_list:
        return 'no'
    elif 'unsure' in ans_word_list:
        return 'unsure'
    else:
        return 'ERROR: '+';'.join(ans_word_list)

def make_prompt(cap, obj, tokenizer):
    _prompt = f'''Here are a few descriptions of an image: {cap}.\nDoes the image contain the following object: {obj}?\nAnswer yes/no/unsure.\n The answer is: '''
    prompt = tokenizer.apply_chat_template([{'role':'user', "content":_prompt}], tokenize=False)
    return prompt

@lru_cache(maxsize=None)
def get_answer(cap, obj, pipe):
    prompt = make_prompt(cap, obj, pipe.tokenizer)
    out = pipe(prompt, max_new_tokens=8, do_sample=False, num_return_sequences=1)
    out = out[0]['generated_text'][len(prompt):].strip()
    out = parse_ans(out)
    return out