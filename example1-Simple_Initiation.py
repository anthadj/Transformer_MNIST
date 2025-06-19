
import torch
from transformer_ah import Transformer_AH 

# Example with custom dimensions
model = Transformer_AH(

    # Encoder variables
    enc_emb_dim=128,
    enc_heads=8,
    enc_layers=2,
    ff_dim=256,
    token_feature_dim=196,  # MUST BE EQUAL to last dimension of enc_in 
                            # i.e. enc_in.shape = (batch, seq_len, token_feature_dim)

    # Decorder variables
    vocab_size=12,
    dec_emb_dim=128,
    dec_heads=8,
    dec_layers=2,

    # Trans-Combine variables
    transformer_emb_dim=128,
    comb_heads=8,
    comb_layers=2,
)

# Dummy inputs
enc_in = torch.randn(4, 6, 196)  # (Batch, seq_len, token_feature_dim)
dec_in = torch.randint(0, 12, (4, 5))  # (Batch, seq_len) 

# How to use only encoder - in training loop
encoder_output = model(enc_in=enc_in, mode="encoder")
print("Encoder output shape ", encoder_output.shape) 

# How to use only decoder (masked attention part) - in training loop
decoder_output = model(dec_in=dec_in, mode = "decoder")
print("Decoder output shape ", decoder_output.shape)

# How to use transformer - in training loop
trans_output = model(enc_in=enc_in,dec_in=dec_in,mode="transformer")  
print("Transformer output shape ", trans_output.shape)