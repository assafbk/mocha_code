from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluate import load
import torch
import numpy as np
import time

from transformers import CLIPProcessor, CLIPModel


def get_reward_model(config, metric):
    if metric == 'bertscore':
        return Bertscore(config)
    elif metric == 'bart_nli':
        return BartNLI(config)
    elif metric == 'deberta_nli':
        return Debertav3NLI(config)
    elif metric == 'clip':
        return Clip(config)
    elif metric == 'meteor':
        return Meteor(config)
    else:
        raise NotImplementedError("reward model not supported")


class reward_model():

    def __init__(self, config):
        self.config = config
        self.device = config['reward_model_device']

    '''
    Important Note: the convention is that the reward is that high reward means 'good caption'
                    (e.g. for NLI p_contradict we need to flip the order of the rewards (because high p_contradict means 'bad performance'))
    '''
    def calculate_rewards(self, img_cap_data, samples, image):
        raise 'calculate_rewards should not be called directly from the base class'
    

'''
type - how to use the nli model. currently support just "nli_p_contradict", which sets the reward to the estimated contradiction probability
'''
class BartNLI(reward_model):
    
    def __init__(self, config, type='nli_p_contradict'):
        super().__init__(config)

        if type not in ['nli_p_contradict']:
            raise '{} type for NLI reward model is not supported'.format(type)
        else:
            self.type = type

        print('Loading BART-NLI model...')
        start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli', cache_dir=config['cache_dir'])
        self.model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli', cache_dir=config['cache_dir'])
        self.model.to(self.device)
        self.model.eval()
        end = time.time()
        print(f'NLI model loaded; {end - start:.2f}s elapsed')
        # number of reference captions per image
        self.num_ref = config['num_ref']


    '''
    Important Note: the convention is that the reward is that high reward means 'good caption'
                    (e.g. for NLI p_contradict we need to flip the order of the rewards (because high p_contradict means 'bad performance'))
    '''
    def calculate_rewards(self, gt_caption, samples, image):
        gt_per_sample = [gt_caption[i] for _ in range(len(samples)) for i in range(self.num_ref)]
        samples = [samples[i] for i in range(len(samples)) for _ in range(self.num_ref)]
        
        with torch.no_grad():
            toks = self.tokenizer(gt_per_sample, samples, return_tensors='pt', padding=True, truncation=True).to(self.device)
            logits = self.model(**toks).logits
            probs = logits.softmax(axis=-1)
            probs = probs.cpu().numpy()

        if self.type == 'nli_p_contradict':
            rewards = 1 - probs[:,0].reshape(-1, self.num_ref).mean(axis=-1) # we flip the order of the rewards because high contradiction prob means bad contradiction score
            rewards = (rewards - 0.5) * 2

        return {'bart_nli' : rewards.tolist()}
            
class Debertav3NLI(reward_model):
    
    def __init__(self, config, type='nli_p_contradict'):
        super().__init__(config)

        if type not in ['nli_p_contradict']:
            raise '{} type for NLI reward model is not supported'.format(type)
        else:
            self.type = type

        print('Loading Deberta-v3 NLI model...')
        start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-large-mnli', cache_dir=config['cache_dir'])
        self.model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli', cache_dir=config['cache_dir'])
        self.model.to(self.device)
        self.model.eval()
        end = time.time()
        print(f'NLI model loaded; {end - start:.2f}s elapsed')
        self.num_ref = config['num_ref'] # number of reference captions per image

    '''
    Important Note: the convention is that the reward is that high reward means 'good caption'
                    (e.g. for NLI p_contradict we need to flip the order of the rewards (because high p_contradict means 'bad performance'))
    '''
    def calculate_rewards(self, gt_caption, samples, image):
        gt_per_sample = [gt_caption[i] for _ in range(len(samples)) for i in range(self.num_ref)]
        samples = [samples[i] for i in range(len(samples)) for _ in range(self.num_ref)]
        
        with torch.no_grad():
            toks = self.tokenizer(gt_per_sample, samples, return_tensors='pt', padding=True, truncation=True).to(self.device)
            logits = self.model(**toks).logits
            probs = logits.softmax(axis=-1)
            probs = probs.cpu().numpy()

        if self.type == 'nli_p_contradict':
            rewards = 1 - probs[:,0].reshape(-1, self.num_ref).mean(axis=-1) # we flip the order of the rewards because high contradiction prob means bad contradiction score
            rewards = (rewards - 0.5) * 2

        return {'deberta_nli' : rewards.tolist()}

class Bertscore(reward_model):
    
    def __init__(self, config):
        super().__init__(config)
        start = time.time()
        print('Loading bertscore model...')
        self.bertscore = load("bertscore", device=config['reward_model_device'])
        self.num_ref = config['num_ref'] # number of reference captions per image
        end = time.time()
        print(f'bertscore model loaded; {end - start:.2f}s elapsed')
        
    def calculate_rewards(self, gt_caption, samples, image):
        bertscores = self.bertscore.compute(predictions=samples,
                                            references=[[gt_caption[i] for i in range(len(gt_caption))] for j in range(len(samples))],
                                            rescale_with_baseline=True,
                                            lang='en',
                                            device=self.device)['f1']
        rewards = np.clip((np.array(bertscores) - 0.35)*2, -1, 1) # normalize between 1 and -1 
        return {'bertscore' : rewards.tolist()}

class Clip(reward_model):
    
    def __init__(self, config):
        super().__init__(config)

        print('Loading CLIP model...')
        start = time.time()
        self.tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=config["cache_dir"])
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=config["cache_dir"])

        self.model.to(self.device)
        self.model.eval()
        end = time.time()
        print(f'CLIP model loaded; {end - start:.2f}s elapsed')

    '''    
    Important Note: the convention is that the reward is that high reward means 'good caption'
                    (e.g. for NLI p_contradict we need to flip the order of the rewards (because high p_contradict means 'bad performance'))
    '''
    def calculate_rewards(self, gt_caption, samples, image):
        
        inputs = self.tokenizer(text=samples, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            rewards = np.squeeze(logits_per_image.detach().cpu().numpy()) / self.model.logit_scale.exp().cpu().numpy()
            rewards = (rewards - 0.3) * 10
        
        return {'clip' : rewards.tolist()}

class Meteor(reward_model):
    
    def __init__(self, config):
        super().__init__(config)
        self.meteor = load("meteor")
        self.num_ref = config['num_ref'] # number of reference captions per image
        
    def calculate_rewards(self, gt_caption, samples, image=None):

        references = [[str(gt_caption[i]) for i in range(self.num_ref)] for _ in range(len(samples))]
        meteor_scores = []
        for reference, sample in zip(references, samples):
            score = self.meteor.compute(predictions=[sample],
                                        references=[reference])['meteor']
            meteor_scores.append(score)

        rewards = np.array(meteor_scores)
        return {'meteor' : rewards}


class GenericRewardModel(reward_model):
    
    def __init__(self, config):
        super().__init__(config)
        self.reward_models = [get_reward_model(config, config["reward_model_list"][i]) for i in range(len(config["reward_model_list"]))]
        self.weights = np.array(config["reward_model_weights"])
        self.average_type = config["reward_model_type"]

        assert len(self.weights) == len(self.reward_models), "number of weights must be identical to number of reward models"

    @staticmethod
    def _multiply_along_axis(A, B, axis):
        return np.swapaxes(np.swapaxes(A, axis, -1) * B, -1, axis)
    
    def _weighted_geometric_mean(self, r):
        nom = np.sum(self._multiply_along_axis(np.log(r), self.weights, 0), axis=0)
        denom = np.sum(self.weights)
        return np.exp(nom/denom)
    
    def _weighted_arithmetic_mean(self, r):
        nom = np.sum(self._multiply_along_axis(r, self.weights, 0), axis=0)
        denom = np.sum(self.weights)
        return nom / denom

    def _get_average(self, rewards_values):
        if self.average_type == 'gm':
            rewards_values =  self._weighted_geometric_mean(rewards_values)
        elif self.average_type == 'am':
            rewards_values =  self._weighted_arithmetic_mean(rewards_values)
        return rewards_values
        
    
    def calculate_rewards(self, samples, gt_caption=None, image=None):
        rewards_list = [rm.calculate_rewards(gt_caption, samples, image) for rm in self.reward_models]
        rewards_dict = {list(rewards_list[i].keys())[0]: list(rewards_list[i].values())[0] for i in range(len(rewards_list))}
        rewards_values = np.array(list(rewards_dict.values()))
        
        if len(self.reward_models) > 1:
            rewards_values = self._get_average(rewards_values)
        else:
            rewards_values = rewards_values[0]
        rewards_dict["reward_list"] = rewards_values.tolist()
        return rewards_dict