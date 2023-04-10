import torch
import clip
from jina import Executor, requests, DocumentArray, Document
from PIL import Image
# from Executors.BaseModel import base_model


class VideoExecutor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('\nLoading CLIP model...')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = 'ViT-B/32'
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print('\nCLIP model loaded.')
        # self.preprocess, self.model = base_model.get_preprocessor_and_model()

    @requests(on='/video_encode')
    def encode(self, docs: DocumentArray, **kwargs):
        with torch.no_grad():
            for doc in docs:
                image_inputs = [self.preprocess(Image.fromarray(chunk.tensor)).to(self.device) for chunk in doc.chunks]
                image_inputs = torch.stack(image_inputs, dim=0)
                print('image_inputs', image_inputs.shape)
                image_features = self.model.encode_image(image_inputs)
                doc.embedding = image_features.cpu().numpy().astype('float32')

                # image_tensors = []
                # for chunk in doc.chunks:
                #     image = Image.fromarray(chunk.tensor)
                #     image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                #     image_features = self.model.encode_image(image_input)
                #     image_tensors.append(image_features.cpu().numpy().astype('float32'))
                # doc.embedding = image_tensors
        print('Embedding done.')
        print('docs', docs)




