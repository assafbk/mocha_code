import torch


'''
the class is a wrapper for a VLM model, which defines an interface for sampling from the VLM
note that the vlm is given to the model, so this class does not initialize the vlm
'''
class vlm_sampler():
    def __init__(self, config, model, model_processor):
        
        self.config = config
        self.device = config['model_device']
        self.model = model
        self.model_processor = model_processor # preprocessor and postprocessor, especially useful for embedding and de-embedding
        self.num_of_samples_per_img = config['num_of_samples_per_image']
    
    def sample_captions_from_model(self):
        raise 'sample_captions_from_model should not be called directly from the base class'



class blip_sampler(vlm_sampler):

    def __init__(self, config, model, model_processor):
        super().__init__(config, model, model_processor)
        self.temperature = config['sampling_temperature']
        
    '''no need for torch.no_grad, it is called inside self.model.generate'''
    def sample_captions_from_model(self, image):
        
        training_state = self.model.training
        if training_state:
            self.model.eval()
        
        inputs = self.model_processor(image, return_tensors="pt").to(self.device)
        outputs_embedded = self.model.generate(**inputs, do_sample=True, top_p=0.9, num_return_sequences=self.num_of_samples_per_img, temperature=self.temperature, max_length=40)
        sampled_captions = []
        for k in range(0,self.num_of_samples_per_img):
            if self.config["reward_model_type"] == 'clip' and len(outputs_embedded[k]) > 77:
                outputs_embedded[k] = outputs_embedded[k][0:76]
            
            sampled_captions.append(self.model_processor.decode(outputs_embedded[k], skip_special_tokens=True))
            sampled_captions = [s.strip() for s in sampled_captions]

        if training_state:
            self.model.train()

        return sampled_captions

    '''no need for torch.no_grad, it is called inside self.model.generate'''
    def greedy_sample(self, image):
        
        training_state = self.model.training
        if training_state:
            self.model.eval()
        
        inputs = self.model_processor(image, return_tensors="pt").to(self.device)
        outputs_embedded = self.model.generate(**inputs, do_sample=False, max_length=40)
        sampled_caption = self.model_processor.decode(outputs_embedded[0], skip_special_tokens=True)

        if training_state:
            self.model.train()

        return sampled_caption

    '''no need for torch.no_grad, it is called inside self.model.generate'''
    def beam_search_sample(self, image):
        
        training_state = self.model.training
        if training_state:
            self.model.eval()
        
        inputs = self.model_processor(image, return_tensors="pt").to(self.device)
        outputs_embedded = self.model.generate(**inputs, max_new_tokens=40, num_beams=5, early_stopping=True)
        sampled_caption = self.model_processor.decode(outputs_embedded[0], skip_special_tokens=True)

        if training_state:
            self.model.train()

        return sampled_caption