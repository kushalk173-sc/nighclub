import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoTokenizer
from .nightclub_ode import NightclubODE
from utils.dev import get_device
import os

# Define constants for our Nightclub analogy
NUM_ROOMS = 4  # MainFloor, VIP_Lounge, QuietBar, OutdoorPatio
VIBE_DIM = 64  # The dimensionality of the "vibe" vector in each room

class FluidNetwork(nn.Module):
    """
    A powerful, multimodal baseline model for the testbed.
    V2: This version uses a Neural ODE with a fluid, attention-based
    graph structure as its core.
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        device = get_device()
        print("Initializing FluidNetwork model (v2 - Attention ODE)...")

        # --- 1. Modality-Specific Encoders ---
        self.vision_encoder = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0).to(device)
        self.audio_encoder = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.text_encoder_name = "distilbert-base-uncased"
        self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)

        # --- 2. Projectors to create the "Initial Vibe" h(0) ---
        # These layers will take the encoder outputs and map them to the initial state
        # of our nightclub's rooms: (batch_size, num_rooms, vibe_dim)
        self.vision_projector = nn.Linear(self.vision_encoder.num_features, NUM_ROOMS * VIBE_DIM)
        # For audio, we'll take the mean of the sequence
        self.audio_projector = nn.Linear(self.audio_encoder.config.hidden_size, NUM_ROOMS * VIBE_DIM)
        # For text, we'll use the [CLS] token's output
        self.text_projector = nn.Linear(self.text_encoder.config.hidden_size, NUM_ROOMS * VIBE_DIM)

        # --- 3. The Core Fluid Dynamics Model ---
        self.nightclub_ode = NightclubODE(VIBE_DIM)

        # --- 4. Task-specific Heads (act on the "Final Vibe") ---
        # The input to these heads is the flattened final state of all rooms
        final_vibe_dim = NUM_ROOMS * VIBE_DIM
        self.vision_head = nn.Linear(final_vibe_dim, num_classes)
        self.asr_head = nn.Linear(final_vibe_dim, self.audio_encoder.config.vocab_size)
        self.text_head = nn.Linear(final_vibe_dim, 2)
        self.regression_head = nn.Linear(final_vibe_dim, 1)

        print("FluidNetwork model initialized successfully.")

    def forward(self, data, pillar_id):
        # --- Step 1: Encode input and create Initial Vibe h(0) ---
        if pillar_id == 1:
            # Note: ASR is complex. For this baseline, we simplify by taking the mean
            # of the audio features to create the initial vibe.
            if data.dim() == 1: data = data.unsqueeze(0)
            encoded_state = self.audio_encoder(data).last_hidden_state.mean(dim=1)
            initial_vibe_flat = self.audio_projector(encoded_state)
        elif pillar_id == 2:
            encoded_state = self.vision_encoder(data)
            initial_vibe_flat = self.vision_projector(encoded_state)
        elif pillar_id in [4, 5]:
            encoded_state = self.text_encoder(**data).last_hidden_state[:, 0] # CLS token
            initial_vibe_flat = self.text_projector(encoded_state)
        else:
            # Lazy initialization of a generic projector for other pillars
            if not hasattr(self, 'generic_projector'):
                device = data.device # Ensure projector is on the same device as data
                self.generic_projector = nn.Linear(data.shape[1], NUM_ROOMS * VIBE_DIM).to(device)
            initial_vibe_flat = self.generic_projector(data)

        # Reshape the flat vector into the structured "vibe" tensor for the ODE
        batch_size = initial_vibe_flat.shape[0]
        initial_vibe = initial_vibe_flat.view(batch_size, NUM_ROOMS, VIBE_DIM)

        # --- Step 2: Evolve the vibe using the Nightclub ODE ---
        # We process each item in the batch individually
        final_vibes_list = [self.nightclub_ode(v) for v in initial_vibe]
        final_vibe = torch.stack(final_vibes_list)
        
        # Flatten the final state to feed into the task heads
        final_vibe_flat = final_vibe.view(batch_size, -1)

        # --- Step 3: Apply the correct task-specific head ---
        if pillar_id == 1:
            logits = self.asr_head(final_vibe_flat)
        elif pillar_id == 2:
            logits = self.vision_head(final_vibe_flat)
        elif pillar_id in [4, 5]:
            logits = self.text_head(final_vibe_flat)
        else:
            logits = self.regression_head(final_vibe_flat)
        
        return logits
            
    def transcribe(self, audio_batch):
        # For now, we return a fixed string for each item in the batch.
        batch_size = audio_batch.shape[0]
        return [f"mock_transcription_for_item_{i}" for i in range(batch_size)]

    def predict(self, data, pillar_id):
        print(f"  - [FluidNetwork] Predicting for Pillar {pillar_id}...")
        
        if pillar_id in [4, 5] and isinstance(data, list):
            data = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # The forward pass now handles all data types
        with torch.no_grad():
            output = self.forward(data, pillar_id)
        
        return output 