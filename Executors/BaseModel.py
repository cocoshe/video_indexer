import clip
import torch


class BaseModel:
    def __init__(self):
        print('\nLoading CLIP model...')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = 'ViT-B/32'
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print('\nCLIP model loaded.')

    def get_preprocessor_and_model(self):
        return self.preprocess, self.model


base_model = BaseModel()