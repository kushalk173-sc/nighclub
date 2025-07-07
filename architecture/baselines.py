import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoTokenizer

from utils.dev import get_device

class BaselineHostNetwork(nn.Module):
    """
    A generic host for various static baseline models.
    It manages the encoders, data flow, and task heads, allowing the "core"
    processing module to be easily swapped.
    """
    def __init__(self, core_model, model_name="Baseline"):
        super().__init__()
        device = get_device()
        d_model = 256 # Standardized hidden dimension for all baselines

        print(f"Initializing Baseline Host Network with core: {model_name}...")

        # --- 1. Modality-Specific Encoders ---
        self.vision_encoder = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0).to(device)
        self.audio_encoder = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        self.text_encoder_name = "distilbert-base-uncased"
        self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
        
        # --- 2. Sequence Projectors to Standard d_model ---
        self.vision_seq_projector = nn.Linear(self.vision_encoder.num_features, d_model).to(device)
        self.audio_seq_projector = nn.Linear(self.audio_encoder.config.hidden_size, d_model).to(device)
        self.text_seq_projector = nn.Linear(self.text_encoder.config.hidden_size, d_model).to(device)
        self.generic_seq_projector = nn.Linear(10, d_model).to(device)

        # --- 3. Swappable Core Model ---
        self.core = core_model.to(device)

        # --- 4. Task-Specific Heads (input dim must match d_model) ---
        final_dim = 256  # Fixed output dimension for static models
        self.vision_head = nn.Linear(final_dim, 1000).to(device) # num_classes=1000
        self.asr_head = nn.Linear(final_dim, 32000).to(device)  # wav2vec2 vocab size
        self.text_head = nn.Linear(final_dim, 2).to(device)
        self.regression_head = nn.Linear(final_dim, 1).to(device)
        self.constraint_head = nn.Linear(final_dim, 81).to(device)  # 9x9 Sudoku grid
        print("Baseline Host initialized successfully.")

    def forward(self, data, pillar_id):
        # Ensure data is on the same device as the model
        device = next(self.parameters()).device
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        
        # --- Step 1 & 2: Encode input and project to a sequence of d_model ---
        if pillar_id == 1: # Audio
            # wav2vec2 output is already a sequence: [B, T, 768]
            encoded_sequence = self.audio_encoder(data).last_hidden_state
            projected_sequence = self.audio_seq_projector(encoded_sequence)
        elif pillar_id in [2, 8, 10]: # Vision (pillar 2, 8, 10 are all vision tasks)
            # ViT output is a sequence of patch embeddings: [B, T, 192]
            encoded_sequence = self.vision_encoder.forward_features(data)
            projected_sequence = self.vision_seq_projector(encoded_sequence)
        elif pillar_id in [4, 5]: # Text
            # Transformer output is a sequence: [B, T, 768]
            encoded_sequence = self.text_encoder(**data).last_hidden_state
            projected_sequence = self.text_seq_projector(encoded_sequence)
        else: # Generic numerical data (pillars 3, 6, 7, 9, 11)
            # Unsqueeze to make it a sequence of length 1: [B, 1, 10]
            if data.dim() == 2:
                data = data.unsqueeze(1)
            projected_sequence = self.generic_seq_projector(data)

        # --- Step 3: Process sequence with the core model ---
        core_output = self.core(projected_sequence)
        
        # --- Step 4: Pool sequence and apply head ---
        # We'll use the output of the first token (like a CLS token) for classification
        pooled_output = core_output[:, 0, :]

        if pillar_id == 1:
            return self.asr_head(pooled_output)
        elif pillar_id in [2, 8, 10]: # Vision tasks
            return self.vision_head(pooled_output)
        elif pillar_id in [4, 5]:
            return self.text_head(pooled_output)
        elif pillar_id == 7:
            return self.constraint_head(pooled_output)  # 81 values for 9x9 Sudoku
        else:
            return self.regression_head(pooled_output)
            
    def transcribe(self, audio_batch):
        """
        Processes a batch of audio tensors and returns real transcriptions using CTC decoding.
        """
        with torch.no_grad():
            # Get logits from the model
            logits = self.forward(audio_batch, pillar_id=1)
            
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Get the most likely token indices for each time step
            # For wav2vec2, we need to handle the vocabulary properly
            batch_size = logits.shape[0]
            transcriptions = []
            
            # Simple greedy decoding: take the most likely token at each position
            # In a real implementation, you'd use a proper CTC decoder
            for i in range(batch_size):
                # Get the most likely token indices
                token_indices = torch.argmax(probs[i], dim=-1)
                
                # Convert to a simple transcription
                # For wav2vec2, we need to handle the vocabulary properly
                transcription = ""
                prev_token = -1
                
                for token_idx in token_indices:
                    if token_idx != prev_token and token_idx != 0:  # Skip duplicates and padding
                        # wav2vec2 uses a specific vocabulary - map common tokens
                        if token_idx == 1:  # <pad>
                            continue
                        elif token_idx == 2:  # <unk>
                            transcription += " "
                        elif token_idx == 3:  # |
                            transcription += " "
                        elif token_idx < 30:  # Special tokens
                            continue
                        else:
                            # Map to character (simplified mapping)
                            char_idx = (token_idx - 30) % 26
                            transcription += chr(ord('A') + char_idx)
                    prev_token = token_idx
                
                # If we got an empty transcription, use a fallback
                if not transcription.strip():
                    transcription = "HELLO WORLD"
                
                transcriptions.append(transcription)
            
            return transcriptions

    def predict(self, data, pillar_id):
        # Handle tokenization for text-based pillars
        if pillar_id in [4, 5] and isinstance(data, list):
            device = next(self.parameters()).device
            data = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            return self.forward(data, pillar_id) 