from jina import Document, DocumentArray, Executor, requests
import torch
import clip
# from Executors.BaseModel import base_model


class TextExecutor(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('\nLoading CLIP model...')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = 'ViT-B/32'
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print('\nCLIP model loaded.')
        # self.preprocess, self.model = base_model.get_preprocessor_and_model()

    @requests(on='/text_encode')
    def encode(self, docs: DocumentArray, **kwargs):
        with torch.no_grad():
            for doc in docs:
                text = clip.tokenize([doc.text]).to(self.device)
                text_features = self.model.encode_text(text)
                doc.embedding = text_features.cpu().numpy().astype('float32')
        print('Text Embedding done.')
        # print('docs', docs)

