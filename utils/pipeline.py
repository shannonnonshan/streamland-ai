"""Pipeline for chaining multiple models"""


class Pipeline:
    """Orchestrate multiple models in sequence."""
    
    def __init__(self):
        self.models = {}
    
    def register_model(self, name, model):
        """Register a model in the pipeline."""
        self.models[name] = model
    
    def transcribe_and_moderate(self, audio_path):
        """Transcribe audio, then moderate the text."""
        if "whisper" not in self.models or "moderation" not in self.models:
            raise ValueError("whisper and moderation models required")
        
        # Step 1: Transcribe
        transcript = self.models["whisper"].transcribe(audio_path)
        
        # Step 2: Moderate
        moderation = self.models["moderation"].moderate_text(transcript)
        
        return {
            "transcript": transcript,
            "moderation": moderation
        }
    
    def transcribe_and_summarize(self, audio_path):
        """Transcribe audio, then summarize."""
        if "whisper" not in self.models or "summarization" not in self.models:
            raise ValueError("whisper and summarization models required")
        
        # Step 1: Transcribe
        transcript = self.models["whisper"].transcribe(audio_path)
        
        # Step 2: Summarize
        summary = self.models["summarization"].summarize(transcript)
        
        return {
            "transcript": transcript,
            "summary": summary
        }
    
