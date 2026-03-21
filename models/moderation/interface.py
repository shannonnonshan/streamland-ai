"""Content Moderation Interface (Detoxify + OpenNSFW2)"""

from models.base import BaseModel


class ModerationModel(BaseModel):
    """Content moderation for text and images."""
    
    def __init__(self, model_path=None, from_hf=True):
        super().__init__("moderation", model_path or "detoxify", from_hf)
    
    def moderate_text(self, text):
        """Check if text is safe. Returns severity scores."""
        raise NotImplementedError("moderate_text() must be implemented")
    
    def moderate_image(self, image_path):
        """Check if image is NSFW safe. Returns scores."""
        raise NotImplementedError("moderate_image() must be implemented")
