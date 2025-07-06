import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoTokenizer
from .v1_graph_ode import V1_GraphODE

# This file restores the v1 architecture top-level model.

NUM_AGENTS = 4
VIBE_DIM = 64

class FluidNetworkV1(nn.Module):
    """
    V1: This version uses a Neural ODE with a fixed graph structure as its core.
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        print("Initializing FluidNetwork model (v1 - Fixed Graph ODE)...")

        # --- Encoders ---
        self.vision_encoder = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        self.audio_encoder = AutoModel.from_pretrained("facebook/wav2vec2-base-960h")
        self.text_encoder_name = "distilbert-base-uncased"
        self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)

        # --- Projectors ---
        self.vision_projector = nn.Linear(self.vision_encoder.num_features, NUM_AGENTS * VIBE_DIM)
        self.audio_projector = nn.Linear(self.audio_encoder.config.hidden_size, NUM_AGENTS * VIBE_DIM)
        self.text_projector = nn.Linear(self.text_encoder.config.hidden_size, NUM_AGENTS * VIBE_DIM)

        # --- Core v1 Model ---
        self.graph_ode = V1_GraphODE(
            vibe_dim=VIBE_DIM,
            num_agents=NUM_AGENTS
        )

        # --- Heads ---
        final_vibe_dim = NUM_AGENTS * VIBE_DIM
        self.vision_head = nn.Linear(final_vibe_dim, num_classes)
        self.asr_head = nn.Linear(final_vibe_dim, self.audio_encoder.config.vocab_size)
        self.text_head = nn.Linear(final_vibe_dim, 2)
        self.regression_head = nn.Linear(final_vibe_dim, 1)

    def forward(self, data, pillar_id):
        if pillar_id == 1:
            if data.dim() == 1: data = data.unsqueeze(0)
            encoded_state = self.audio_encoder(data).last_hidden_state.mean(dim=1)
            initial_vibe_flat = self.audio_projector(encoded_state)
        elif pillar_id == 2:
            encoded_state = self.vision_encoder(data)
            initial_vibe_flat = self.vision_projector(encoded_state)
        elif pillar_id in [4, 5]:
            encoded_state = self.text_encoder(**data).last_hidden_state[:, 0]
            initial_vibe_flat = self.text_projector(encoded_state)
        else:
            if not hasattr(self, 'generic_projector'):
                self.generic_projector = nn.Linear(10, NUM_AGENTS * VIBE_DIM).to(data.device)
            initial_vibe_flat = self.generic_projector(data)

        batch_size = initial_vibe_flat.shape[0]
        initial_vibe = initial_vibe_flat.view(batch_size, NUM_AGENTS, VIBE_DIM)

        final_vibes_list = [self.graph_ode(v) for v in initial_vibe]
        final_vibe = torch.stack(final_vibes_list)
        
        final_vibe_flat = final_vibe.view(batch_size, -1)

        if pillar_id == 1:
            return self.asr_head(final_vibe_flat)
        elif pillar_id == 2:
            return self.vision_head(final_vibe_flat)
        elif pillar_id in [4, 5]:
            return self.text_head(final_vibe_flat)
        else:
            return self.regression_head(final_vibe_flat)
            
    def transcribe(self, audio_data):
        with torch.no_grad():
            logits = self.forward(audio_data, pillar_id=1)
        return f"vibe_transcription_ids_{torch.argmax(logits, dim=-1).shape}"

    def predict(self, data, pillar_id):
        if pillar_id in [4, 5] and isinstance(data, list):
            data = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            return self.forward(data, pillar_id) 