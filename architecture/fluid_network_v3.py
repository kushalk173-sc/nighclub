import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoTokenizer
from .v3_attention_ode import V3_AttentionODE
from utils.dev import get_device

# Define constants for the v3 model
NUM_AGENTS = 4  # The number of "people" or neurons in the club
VIBE_DIM = 64   # The dimensionality of the "vibe" vector
ROLE_DIM = 16   # The dimensionality of the learned "role" embedding

class FluidNetworkV3(nn.Module):
    """
    A powerful, multimodal baseline model for the testbed.
    V3: This version uses a Neural ODE with affinity-biased attention.
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        device = get_device()
        print("Initializing FluidNetwork model (v3 - Affinity-Biased Attention ODE)...")

        # --- 1. Modality-Specific Encoders ---
        self.vision_encoder = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0).to(device)
        self.audio_encoder = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.text_encoder_name = "distilbert-base-uncased"
        self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)

        # --- 2. Projectors to create the "Initial Vibe" h(0) ---
        self.vision_projector = nn.Linear(self.vision_encoder.num_features, NUM_AGENTS * VIBE_DIM)
        self.audio_projector = nn.Linear(self.audio_encoder.config.hidden_size, NUM_AGENTS * VIBE_DIM)
        self.text_projector = nn.Linear(self.text_encoder.config.hidden_size, NUM_AGENTS * VIBE_DIM)

        # --- 3. The Core Fluid Dynamics Model (v3) ---
        self.attention_ode = V3_AttentionODE(
            vibe_dim=VIBE_DIM,
            role_dim=ROLE_DIM,
            num_agents=NUM_AGENTS
        )

        # --- 4. Task-specific Heads (act on the "Final Vibe") ---
        final_vibe_dim = NUM_AGENTS * VIBE_DIM
        self.vision_head = nn.Linear(final_vibe_dim, num_classes)
        self.asr_head = nn.Linear(final_vibe_dim, self.audio_encoder.config.vocab_size)
        self.text_head = nn.Linear(final_vibe_dim, 2)
        self.regression_head = nn.Linear(final_vibe_dim, 1)

        print("FluidNetwork v3 model initialized successfully.")

    def forward(self, data, pillar_id):
        # --- Step 1: Encode input and create Initial Vibe h(0) ---
        if pillar_id == 1:
            if data.dim() == 1: data = data.unsqueeze(0)
            encoded_state = self.audio_encoder(data).last_hidden_state.mean(dim=1)
            initial_vibe_flat = self.audio_projector(encoded_state)
        elif pillar_id == 2:
            encoded_state = self.vision_encoder(data)
            initial_vibe_flat = self.vision_projector(encoded_state)
        elif pillar_id in [4, 5]:
            encoded_state = self.text_encoder(**data).last_hidden_state[:, 0] # CLS token
            initial_vibe_flat = self.text_projector(encoded_state)
        else: # Generic numerical data
            if not hasattr(self, 'generic_projector'):
                device = data.device # Ensure projector is on the same device as data
                self.generic_projector = nn.Linear(data.shape[1], NUM_AGENTS * VIBE_DIM).to(device)
            initial_vibe_flat = self.generic_projector(data)

        batch_size = initial_vibe_flat.shape[0]
        initial_vibe = initial_vibe_flat.view(batch_size, NUM_AGENTS, VIBE_DIM)

        # --- Step 2: Evolve the vibe using the v3 Attention ODE ---
        final_vibes_list = [self.attention_ode(v) for v in initial_vibe]
        final_vibe = torch.stack(final_vibes_list)
        
        final_vibe_flat = final_vibe.view(batch_size, -1)

        # --- Step 3: Apply the correct task-specific head ---
        if pillar_id == 1:
            logits = self.asr_head(final_vibe_flat)
        elif pillar_id == 2:
            logits = self.vision_head(final_vibe_flat)
        elif pillar_id in [4, 5]:
            logits = self.text_head(final_vibe_flat)
        else:
            return self.regression_head(final_vibe_flat)
        
        return logits
            
    def transcribe(self, audio_batch):
        """
        Processes a batch of audio tensors and returns a batch of mock transcriptions.
        """
        with torch.no_grad():
            logits = self.forward(audio_batch, pillar_id=1)
        
        # This is a mock transcription. In a real scenario, this would involve
        # a proper decoding algorithm (e.g., CTC beam search).
        # For now, we return a fixed string for each item in the batch.
        batch_size = audio_batch.shape[0]
        return [f"mock_transcription_for_item_{i}" for i in range(batch_size)]

    def predict(self, data, pillar_id):
        if pillar_id in [4, 5] and isinstance(data, list):
            data = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            output = self.forward(data, pillar_id)
        
        return output 