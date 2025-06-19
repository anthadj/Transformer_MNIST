# Transformer_AH

A custom Transformer model implemented in PyTorch, designed for flexibility in encoder-decoder architectures.  
Includes separate modules for:
- Encoder
- Masked decoder
- Transformer - Encoder-Masked decoder combiner

Supports easy switching between encoder-only, decoder-only, and full transformer modes.

---

## 📂 Example files

**`example1-Simple_Initiation.py`**  
A minimal example showing how to:
- Instantiate the model
- Run data through encoder, decoder, or full transformer modes

**`example2-MNIST_Transformer.py`**  
A more complete example where:
- MNIST image data is processed
- The transformer model is trained on MNIST as input
- Demonstrates how to integrate the model into a training loop

---

## Instantiate the model

```


from transformer_ah import Transformer_AH

model = Transformer_AH(
    enc_emb_dim=128,
    enc_heads=8,
    enc_layers=2,
    ff_dim=256,
    token_feature_dim=196,
    vocab_size=12,
    dec_emb_dim=128,
    dec_heads=8,
    dec_layers=2,
    transformer_emb_dim=128,
    comb_heads=8,
    comb_layers=2
)

```



## Run encoder, decoder, or transformer modes

```

import torch

# Example encoder input: (batch_size, seq_len, token_feature_dim)
enc_in = torch.randn(4, 6, 196)

# Example decoder input: (batch_size, seq_len)
dec_in = torch.randint(0, 12, (4, 5))

# Encoder only
enc_out = model(enc_in=enc_in, mode="encoder")

# Decoder only
dec_out = model(dec_in=dec_in, mode="decoder")

# Full transformer
full_out = model(enc_in=enc_in, dec_in=dec_in, mode="transformer")

```


Example commands
Run the simple initiation example:

```
python example1-Simple_Initiation.py
```

Run the MNIST transformer example:

```
python example2-MNIST_Transformer.py
```

Ensure that your token_feature_dim matches the encoder input’s last dimension. 
