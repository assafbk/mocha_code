import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import os
import shutil
from datetime import datetime
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from torch.utils.data import Dataset

import vlm_sampler as vlm_sampler_module
import reward_model as reward_model_module
from dataset import *
from utils import *

def save_model(config, config_file, model, model_processor, step, start_time, best_model=False):
    rm_str = '_'.join(f"{str(w)}_{str(r)}" for r, w in zip(config["reward_model_list"], config["reward_model_weights"]))
    output_dir = os.path.join(config['output_dir'], f'{start_time}_{config["model_type"]}_{config["dataset"]}_{config["reward_model_type"]}_{rm_str}_reward_model_{config["num_ref"]}_refs_seed_{config["seed"]}/')
    if best_model:
        ckpt = os.path.join(output_dir, f'best_model')
    else:
        ckpt = os.path.join(output_dir, f'step_{step}')
    model.save_pretrained(ckpt)
    model_processor.save_pretrained(ckpt)
    shutil.copy(config_file, os.path.join(ckpt,'vlm_rlhf_config.json'))

def set_ref_model_to_inference_mode(ref_model):
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

def get_local_coco_data_loaders(config):
    train_annotations_path = config["coco_annotations_train"]
    val_annotations_path = config["coco_annotations_val"]
    data_train = pd.read_parquet(train_annotations_path)
    data_val = pd.read_parquet(val_annotations_path)
    
    path_to_coco_images = config['path_to_coco_images']
    dataset_train = COCO(data=data_train, images_path=os.path.join(path_to_coco_images,'train2014'))
    dataset_val = COCO(data=data_val.iloc[0:20], images_path=os.path.join(path_to_coco_images,'val2014'))
    data_loader_train = DataLoader(dataset_train, collate_fn=local_coco_collate_fn, batch_size=config['num_of_images_per_batch'], shuffle=True, num_workers=0) # Note: the effective batch size will be: <num of gt captions> x <num of generated captions per gt caption>
    data_loader_val = DataLoader(dataset_val, collate_fn=local_coco_collate_fn, batch_size=1, shuffle=False, num_workers=0) # we have 20 images in the val set and we evaluate them one by one, hence batch_size=1
    return data_loader_train, data_loader_val
    
def get_url_data_loaders_coco(config):
    USER_AGENT = get_datasets_user_agent()
    dataset = load_dataset("yerevann/coco-karpathy", cache_dir=config['cache_dir'])
    dataset_smaller_val_split = dataset['validation'].train_test_split(test_size=20/dataset['validation'].num_rows, seed=111) # we set the same seed because we do want to shuffle the dataset before selecting the val set, but want it to be consistent every time.
    data_loader_train = DataLoader(dataset['train'], collate_fn=url_coco_collate_fn, batch_size=config['num_of_images_per_batch'], shuffle=True, num_workers=0) # Note: the effective batch size will be: <num of gt captions> x <num of generated captions per gt caption>
    data_loader_val = DataLoader(dataset_smaller_val_split['test'], collate_fn=url_coco_collate_fn, batch_size=1, shuffle=False, num_workers=0) # we have 20 images in the val set and we evaluate them one by one, hence batch_size=1
    
    return data_loader_train, data_loader_val

def get_data_loaders(config):
    USER_AGENT = get_datasets_user_agent()
    dataset = load_dataset("nlphuji/flickr30k", cache_dir=config['cache_dir'])
    dataset = dataset['test'] # seems like all of the data is here...
    dataset = dataset.train_test_split(test_size=20/dataset.num_rows, seed=111) # we set the same seed because we do want to shuffle the dataset before selecting the val set, but want it to be consistent every time.
    data_loader_train = DataLoader(dataset['train'], collate_fn=collate_fn, batch_size=config['num_of_images_per_batch'], shuffle=True, num_workers=0) # Note: the effective batch size will be: <num of gt captions> x <num of generated captions per gt caption>
    data_loader_val = DataLoader(dataset['test'], collate_fn=collate_fn, batch_size=1, shuffle=False, num_workers=0) # we have 20 images in the val set and we evaluate them one by one, hence batch_size=1
    
    return data_loader_train, data_loader_val

def freeze_vision_encoder(model):
    for param in model.vision_model.parameters():
        param.requires_grad = False

def load_vlm(config):
    
    model_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large', cache_dir=config['cache_dir'])
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large', cache_dir=config['cache_dir']).to(config['model_device'])
    
    if config['freeze_vision_encoder']:
        freeze_vision_encoder(model)

    return model_processor, model

def load_reward_model(config):
    reward_model = reward_model_module.GenericRewardModel(config)
    return reward_model

'''
the function gets a batch of img_cap_data, and for each image:
1. samples the VLM
2. calculates rewards for the samples, w.r.t the ground truth
3. composes the rlhf_batch for rlhf training
   (rlhf_batch stores the samples in lists (e.g. on list for images, one list for vlm samples, etc))
'''
def gen_data_for_batch(batch, vlm_sampler, reward_model):
    
    rlhf_batch = {}
    rlhf_batch['image_list'] = []
    rlhf_batch['gt_caption_list'] = []
    rlhf_batch['vlm_sample_list'] = []

    # used by REINFORCE (SCST)
    rlhf_batch['vlm_baseline_list'] = []
    rlhf_batch['baselined_reward_list'] = []

    for img_cap_data in batch:
        rlhf_batch['image_list'].extend([img_cap_data['image'] for _ in range(config['num_of_samples_per_image'])])
        rlhf_batch['gt_caption_list'].extend([[img_cap_data['caption'][i] 
                                               for i in range(len(img_cap_data['caption']))] 
                                               for _ in range(config['num_of_samples_per_image'])])

        samples_for_cur_image = vlm_sampler.sample_captions_from_model(img_cap_data['image'])
        rlhf_batch['vlm_sample_list'].extend(samples_for_cur_image)
        reward_list_cur_samples = reward_model.calculate_rewards(samples_for_cur_image, img_cap_data['caption'], img_cap_data['image'],)
        
        # add all reward metrics
        for reward_name, reward in reward_list_cur_samples.items():
            if reward_name not in rlhf_batch:
                rlhf_batch[reward_name] = reward
            else:
                rlhf_batch[reward_name].extend(reward)

        # used by REINFORCE (SCST)
        # inference_policy_sample_for_cur_image = vlm_sampler.beam_search_sample(img_cap_data['image'])
        # inference_policy_sample_reward = reward_model.calculate_rewards(img_cap_data['caption'][0], [inference_policy_sample_for_cur_image])

        mean_cur_samples = np.mean(reward_list_cur_samples['reward_list'])
        rlhf_batch['vlm_baseline_list'].extend([' ' for _ in range(config['num_of_samples_per_image'])])
        rlhf_batch['baselined_reward_list'].extend([sampled_reward-mean_cur_samples for sampled_reward in reward_list_cur_samples['reward_list']])

    return rlhf_batch

def calc_rlhf_loss(config, predicted_logits, old_predicted_logits, ref_predicted_logits, vlm_samples_encoded, rewards):
    
    batch_size = predicted_logits.shape[0]
    clipped_loss = 0
    total_kl_for_metric = 0
    kl_reward_penalty_per_sample = []
    for idx, vlm_sample in enumerate(vlm_samples_encoded):
        
        labels = vlm_sample[1:]
        cur_mask = (labels > 0).to(predicted_logits)
        cur_num_of_tokens = np.sum(cur_mask.int().cpu().tolist())
        
        labels = labels[:cur_num_of_tokens]
        cur_sample_predicted_logits = predicted_logits[idx,:cur_num_of_tokens,:]
        cur_sample_old_predicted_logits = old_predicted_logits[idx,:cur_num_of_tokens,:]
        cur_sample_ref_predicted_logits = ref_predicted_logits[idx,:cur_num_of_tokens,:]

        sampled_words_probability = torch.exp(cur_sample_predicted_logits[torch.arange(cur_num_of_tokens),labels])/torch.exp(cur_sample_predicted_logits).sum(dim=1)
        old_sampled_words_probability = torch.exp(cur_sample_old_predicted_logits[torch.arange(cur_num_of_tokens),labels])/torch.exp(cur_sample_old_predicted_logits).sum(dim=1)
        ref_sampled_words_probability = torch.exp(cur_sample_ref_predicted_logits[torch.arange(cur_num_of_tokens),labels])/torch.exp(cur_sample_ref_predicted_logits).sum(dim=1)
        
        cur_kl_est = torch.sum(torch.log(sampled_words_probability.detach().clone())) - torch.sum(torch.log(ref_sampled_words_probability)) # log(pi_rl(vlm_sample)/pi_sft(vlm_sample)). this is a 'single sample' estimation of KL between pi_rl and pi_phi (reminder: we sample from the pi_rl dist.)
        cur_kl_penalty_term = (-1) * config['beta'] * cur_kl_est
        cur_reward = rewards[idx] + cur_kl_penalty_term 
        kl_reward_penalty_per_sample.append(cur_kl_penalty_term.cpu())

        ratio = sampled_words_probability/old_sampled_words_probability
        surr1 = ratio * cur_reward
        surr2 = torch.clamp(ratio, 1 - config['epsilon'], 1 + config['epsilon']) * cur_reward
        clipped_loss += -torch.min(surr1, surr2).mean()
        
        # for metric
        total_kl_for_metric += cur_kl_est

    rlhf_loss = clipped_loss / batch_size
    total_kl_for_metric = total_kl_for_metric / batch_size

    return rlhf_loss, total_kl_for_metric, kl_reward_penalty_per_sample

def evaluate_validation_set(vlm_sampler, reward_model, data_loader_val, config, cur_step):
    
    samples_df = pd.DataFrame()
    val_log = {}
    reward_means = []
    reward_stds = []
    nli_means = []
    bertscore_means = []
    clip_means = []
    for img_idx, batch in enumerate(data_loader_val):
        rlhf_batch_val = gen_data_for_batch(batch, vlm_sampler, reward_model)
        cur_samples_df = pd.DataFrame(rlhf_batch_val)
        cur_samples_df.insert(loc=0, column='step', value=[idx for _ in range(len(rlhf_batch_val['gt_caption_list']))])
        cur_samples_df.insert(loc=1, column='val_img_idx', value=[img_idx for _ in range(len(rlhf_batch_val['gt_caption_list']))])
        cur_samples_df['sample_idx_per_img'] =  list(range(config['num_of_samples_per_image']))
        cur_samples_df.drop('image_list', axis=1, inplace=True)
        samples_df = samples_df._append(cur_samples_df, ignore_index=True)

        reward_mean_cur_image = np.mean(rlhf_batch_val['reward_list'])
        reward_std_cur_image = np.std(rlhf_batch_val['reward_list'])

        if 'bart_nli' in config['reward_model_list']:
            nli_reward_mean_cur_batch = np.mean(rlhf_batch_val['bart_nli'])
            nli_means.append(nli_reward_mean_cur_batch)
        
        if 'deberta_nli' in config['reward_model_list']:
            nli_reward_mean_cur_batch = np.mean(rlhf_batch_val['deberta_nli'])
            nli_means.append(nli_reward_mean_cur_batch)
        
        if 'bertscore' in config['reward_model_list']:
            bertscore_reward_mean_cur_batch = np.mean(rlhf_batch_val['bertscore'])
            bertscore_means.append(bertscore_reward_mean_cur_batch)
        
        if 'clip' in config['reward_model_list']:
            clip_reward_mean_cur_batch = np.mean(rlhf_batch_val['clip'])
            clip_means.append(clip_reward_mean_cur_batch)

        reward_means.append(reward_mean_cur_image)
        reward_stds.append(reward_std_cur_image)
        val_log[f'validation_per_image_reward_mean/image {img_idx}'] = reward_mean_cur_image
        val_log[f'validation_per_image_reward_std/image {img_idx}'] = reward_std_cur_image
    
    total_reward_mean = np.mean(reward_means)
    total_reward_std = np.sqrt(np.mean(np.array(reward_stds)**2))
    val_log[f'validation_reward_mean'] = total_reward_mean
    val_log[f'validation_reward_std'] = total_reward_std
    
    if 'bart_nli' in config['reward_model_list'] or 'deberta_nli' in config['reward_model_list']:
        total_nli_mean = np.mean(nli_means)
    else:
        total_nli_mean = 0

    if 'bertscore' in config['reward_model_list']:
        total_bertscore_mean = np.mean(bertscore_means)
    else:
        total_bertscore_mean = 0
    
    if 'clip' in config['reward_model_list']:
        total_clip_mean = np.mean(clip_means)
    else:
        total_clip_mean = 0
    
    val_log[f'validation_nli_mean'] = total_nli_mean
    val_log[f'validation_bertscore_mean'] = total_bertscore_mean
    val_log[f'validation_clip_mean'] = total_clip_mean

    print(f'\nValidation Set - Epoch: {epoch} | Step: {idx + 1} | Reward Mean: {total_reward_mean:.3f} | Reward STD: {total_reward_std:.3f} | NLI Mean: {total_nli_mean:.3f} | BERTscore Mean: {total_bertscore_mean:.3f} | CLIP Mean: {total_clip_mean:.3f}\n')
        
    return samples_df, val_log

if __name__ == '__main__':

    vlm_rlhf_config_file_name = './vlm_rlhf_config.json'
    config = load_config(vlm_rlhf_config_file_name)
    set_seed(config['seed'])
    if config['activate_logging']:
        wandb.login()
        wandb_run = wandb.init(project="vlm_rlhf", config=config)
    model_processor, model = load_vlm(config)
    model_sampler = vlm_sampler_module.blip_sampler(config, model, model_processor)

    if config['ref_model_device'] is not None:
        ref_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large', cache_dir=config['cache_dir']).to(config['ref_model_device'])
        set_ref_model_to_inference_mode(ref_model)
    else:
        ref_model = None

    reward_model = load_reward_model(config)
    if config['dataset'] == 'flickr':
        data_loader_train, data_loader_val = get_data_loaders(config)
    elif config['dataset'] == 'coco':
        if config['data_loading_type'] == 'local':
            data_loader_train, data_loader_val = get_local_coco_data_loaders(config)
        elif config['data_loading_type'] == 'url':
            data_loader_train, data_loader_val = get_url_data_loaders_coco(config)
        else:
            raise('bad data loading type, please selecet "url"/"local"')
    else:
        raise NotImplementedError
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    start_datetime_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    model.train()
    best_reward = -1
    for idx, batch in enumerate(tqdm(data_loader_train)):
        try:
            if idx > config["max_step"]:
                break
            epoch = float("{:.3f}".format(float("{:.3f}".format(idx / len(data_loader_train)))))
            log_cur_step = {}
            rlhf_batch = gen_data_for_batch(batch, model_sampler, reward_model)
            
            # PPO self-critical update
            inputs = model_processor(images=rlhf_batch['image_list'], text=rlhf_batch['vlm_sample_list'], return_tensors="pt", padding=True)
            pixel_values = inputs['pixel_values'].to(config['model_device'])
            input_tokens = inputs['input_ids'].to(config['model_device'])
            attention_mask = inputs['attention_mask'].to(config['model_device']) # we add attention masks because sometimes the padded data can create wrong outputs. see https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked 
            
            with torch.no_grad():
                out_old = model(input_ids=input_tokens, pixel_values=pixel_values, labels=None, attention_mask=attention_mask)
                old_predicted_logits = out_old.logits
                out_ref = ref_model(input_ids=input_tokens.to(config['ref_model_device']), pixel_values=pixel_values.to(config['ref_model_device']), labels=None, attention_mask=attention_mask.to(config['ref_model_device']))
                ref_predicted_logits = out_ref.logits
                ref_predicted_logits = ref_predicted_logits.to(config['model_device'])
                
            for ppo_iter in range(config["ppo_iters"]):
                out = model(input_ids=input_tokens, pixel_values=pixel_values, labels=None, attention_mask=attention_mask) # this is the way to perform MLM (Masked Language Modeling) with BlipForConditionalGeneration            
                predicted_logits = out.logits
                rlhf_loss, kl_loss, kl_reward_penalty_per_sample = calc_rlhf_loss(config, predicted_logits, old_predicted_logits, ref_predicted_logits, input_tokens, rlhf_batch['baselined_reward_list'])
                optimizer.zero_grad()
                if ppo_iter < config['ppo_iters'] - 1:
                    rlhf_loss.backward(retain_graph=True)
                else:
                    rlhf_loss.backward()
                
                if config['clip_grad']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_max_norm'])
                optimizer.step()

            # metrics
            rlhf_batch['rlhf_reward_list'] = np.array(rlhf_batch['reward_list']) - np.array(kl_reward_penalty_per_sample)
            reward_mean_cur_batch = np.mean(rlhf_batch['reward_list'])
            reward_std_cur_batch = np.std(rlhf_batch['reward_list'])

            if 'bart_nli' in config['reward_model_list']:
                nli_reward_mean_cur_batch = np.mean(rlhf_batch['bart_nli'])
                nli_reward_std_cur_batch = np.std(rlhf_batch['bart_nli'])
            elif 'deberta_nli' in config['reward_model_list']:
                nli_reward_mean_cur_batch = np.mean(rlhf_batch['deberta_nli'])
                nli_reward_std_cur_batch = np.std(rlhf_batch['deberta_nli'])
            else:
                nli_reward_mean_cur_batch = 0
                nli_reward_std_cur_batch = 0

            if 'bertscore' in config['reward_model_list']:
                bertscore_reward_mean_cur_batch = np.mean(rlhf_batch['bertscore'])
                bertscore_reward_std_cur_batch = np.std(rlhf_batch['bertscore'])
            else:
                bertscore_reward_mean_cur_batch = 0
                bertscore_reward_std_cur_batch = 0
            
            if 'clip' in config['reward_model_list']:
                clip_reward_mean_cur_batch = np.mean(rlhf_batch['clip'])
                clip_reward_std_cur_batch = np.std(rlhf_batch['clip'])
            else:
                clip_reward_mean_cur_batch = 0
                clip_reward_std_cur_batch = 0

            baselined_reward_mean_cur_batch = np.mean(rlhf_batch['baselined_reward_list'])
            baselined_reward_std_cur_batch = np.std(rlhf_batch['baselined_reward_list'])
            rlhf_reward_mean_cur_batch = np.mean(rlhf_batch['rlhf_reward_list'])
            rlhf_reward_std_cur_batch = np.std(rlhf_batch['rlhf_reward_list'])
            grad_norm = calc_grad_norm(model)
            mean_entropy_cur_batch = calc_mean_entropy(predicted_logits.detach().clone().cpu())
            
            print(f'Epoch: {epoch} | Step: {idx + 1} | Loss: {rlhf_loss:.3f} | Reward Mean: {reward_mean_cur_batch:.3f} | Reward STD: {reward_std_cur_batch:.3f} | Baselined Reward Mean: {baselined_reward_mean_cur_batch:.3f} | Baselined Reward STD: {baselined_reward_std_cur_batch:.3f}, RLHF Reward Mean: {rlhf_reward_mean_cur_batch:.3f} | RLHF Reward STD: {rlhf_reward_std_cur_batch:.3f}, | Grad Norm: {grad_norm:.3f} | Mean Entropy: {mean_entropy_cur_batch:.3f} | RLHF Loss: {rlhf_loss:.3e} | KL Dist: {kl_loss:.3e} | NLI Reward Mean: {nli_reward_mean_cur_batch:.3f} | NLI Reward STD: {nli_reward_std_cur_batch:.3f} | BERTscore Reward Mean: {bertscore_reward_mean_cur_batch:.3f} | BERTscore Reward STD: {bertscore_reward_std_cur_batch:.3f} | CLIP Reward Mean: {clip_reward_mean_cur_batch:.3f} | CLIP Reward STD: {clip_reward_std_cur_batch:.3f}')
            log_cur_step = {"rlhf_loss": rlhf_loss, "reward_mean": reward_mean_cur_batch, "reward_std": reward_std_cur_batch, "baselined_reward_mean": baselined_reward_mean_cur_batch, "baselined_reward_std": baselined_reward_std_cur_batch, "rlhf_reward_mean": rlhf_reward_mean_cur_batch, "rlhf_reward_std": rlhf_reward_std_cur_batch, 'grad_norm': grad_norm, 'mean_entropy': mean_entropy_cur_batch, 'rlhf_loss': rlhf_loss, 'kl_dist': kl_loss, 'nli_reward_mean': nli_reward_mean_cur_batch, 'nli_reward_std': nli_reward_std_cur_batch, 'bertscore_reward_mean': bertscore_reward_mean_cur_batch, 'bertscore_reward_std': bertscore_reward_std_cur_batch, 'clip_reward_mean': clip_reward_mean_cur_batch, 'clip_reward_std': clip_reward_std_cur_batch} 
            
            if idx % 1 == 0:
                print('\nground truth captions:')
                print(rlhf_batch['gt_caption_list'][0])
                print('\nvlm samples (train):')
                print(*(rlhf_batch['vlm_sample_list'][0:5]), sep='\n')
                print('\n')
            
            # log validation samples, train samples and train data 
            if idx % config['eval_steps'] == 0:

                cur_samples_df = pd.DataFrame(rlhf_batch)
                cur_samples_df.insert(loc=0, column='step', value=[idx for _ in range(len(rlhf_batch['gt_caption_list']))])
                cur_samples_df['sample_idx_per_img'] =  list(range(config['num_of_samples_per_image'])) * config['num_of_images_per_batch']
                cur_samples_df.drop('image_list', axis=1, inplace=True)
                cur_samples_df.drop('rlhf_reward_list', axis=1, inplace=True)
                
                # only log 5 training samples 
                cur_samples_df = cur_samples_df.iloc[:config['num_ref'] * 5].copy()
                cur_samples_df['gt_caption_list'] = [" | ".join(cur_samples_df['gt_caption_list'][idx]) for idx in range(len(cur_samples_df['gt_caption_list']))]
                df_train = df_train._append(cur_samples_df,ignore_index=True)
                
                model.eval()
                cur_df_val, val_log_cur_step = evaluate_validation_set(model_sampler, reward_model, data_loader_val, config, idx)
                cur_reward = val_log_cur_step['validation_reward_mean']
                
                # only log 5 evaluation samples
                cur_df_val = cur_df_val.iloc[:config['num_ref'] * 5].copy()
                cur_df_val['gt_caption_list'] = [" | ".join(cur_df_val['gt_caption_list'][idx]) for idx in range(len(cur_df_val['gt_caption_list']))]
                df_val = df_val._append(cur_df_val,ignore_index=True)
                model.train()
                if config['activate_logging']:
                    log_all = {**log_cur_step, **val_log_cur_step}
                    
                    wandb_table_train = wandb.Table(data=df_train)
                    log_all['train_data'] = wandb_table_train
                    
                    wandb_table_val = wandb.Table(data=df_val)
                    log_all['validation_data'] = wandb_table_val
                    
                    wandb.log(log_all)
                
                # save best model
                if cur_reward > best_reward:
                    best_reward = cur_reward
                    save_model(config, vlm_rlhf_config_file_name, model , model_processor, step=idx, start_time=start_datetime_str, best_model=True)
            # log train data 
            else:
                if config['activate_logging']:
                    wandb.log(log_cur_step)
            # save model
            if idx % config['save_steps'] == 0 and idx != 0:
                save_model(config, vlm_rlhf_config_file_name, model , model_processor, start_time=start_datetime_str,step=idx)
            
        except Exception as e:
            print(e)
            continue
