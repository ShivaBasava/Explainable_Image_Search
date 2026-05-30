import io, json, requests, torch
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
from transformers import AutoProcessor, AutoModel

from torchvision import transforms
import torch.nn.functional as F

from SearchArtWorks import SearchArtWorks
from ArtIndexer import ArtIndexer


from config import get_config
WIKI_HEADERS = get_config("WIKI_HEADERS")
TIMEOUT      = get_config("TIMEOUT")

MODEL_ID = get_config("MODEL_ID")
MAX_LENGTH = 64


class ArtEmbedd:
    """embedding  via Model ( vision and text encoders).
    L2-normalised vectors are returned."""
    MODEL_ID = MODEL_ID
    HEADERS = WIKI_HEADERS 
    TIMEOUT = TIMEOUT
    def __init__(self):

        self.searcher = SearchArtWorks()
        self.art_indexer = ArtIndexer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModel.from_pretrained(self.MODEL_ID).eval()
        self.model.to(self.device)


    def _to_tensor(self, outputs):
    """checking for available output parameters"""
        if isinstance(outputs,
                 torch.Tensor):
            return outputs
        if hasattr(outputs, "pooler_output") and \
            (outputs.pooler_output is not None):
            return outputs.pooler_output
        if hasattr(outputs, "image_embeds") and \
            (outputs.image_embeds is not None):
            return outputs.image_embeds
        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            return outputs.text_embeds
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state.mean(dim=1)
        raise TypeError("Could not extract embedding tensor from model output.")


    def get_embedding(self, input_data) -> np.ndarray:
        """siglip2-naflex"""
        is_image = isinstance(input_data, Image.Image)

        if is_image:
            image = input_data.convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                if hasattr(self.model, "get_image_features"):
                    outputs = self.model.get_image_features(**inputs)
                else:
                    outputs = self.model(pixel_values=inputs["pixel_values"])
        else:
            input_text = input_data if isinstance(input_data, list) else [str(input_data).lower()]
            
            inputs = self.processor(text=input_text, return_tensors="pt", 
                            padding="max_length", max_length=MAX_LENGTH, truncation=True).to(self.device)
            with torch.no_grad():
                if hasattr(self.model, "get_text_features"):
                    outputs = self.model.get_text_features(**inputs)
                else:
                    outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        tensor = self._to_tensor(outputs)

        tensor = torch.nn.functional.normalize(tensor.float(), p=2, dim=-1) # hereim perfoming L2 norm
        return tensor.squeeze(0).cpu().numpy().astype(np.float32) 


    def get_embedding__(self, input_data) -> np.ndarray:
        """
        #tipvs
        """
        is_image = isinstance(input_data, Image.Image)
 
        with torch.no_grad():
            if is_image:

                pixel_values = _IMAGE_TRANSFORM(
                    input_data.convert("RGB")
                ).unsqueeze(0).to(self.device)   # shape: (1, 3, 448, 448)
 
                out = self.model.encode_image(pixel_values)

                tensor = out.cls_token.squeeze(1)
 
            else:

                text_list = [str(input_data).lower()]
                tensor = self.model.encode_text(text_list)
                tensor = tensor.to(self.device)
 
        # L2 normalie
        tensor = F.normalize(tensor.float(), p=2, dim=-1)
        return tensor.squeeze(0).cpu().numpy().astype(np.float32)
