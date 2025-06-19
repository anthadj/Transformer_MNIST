# transformer_ah.py

import torch
import torch.nn.functional as F
import math

class Transformer_AH(torch.nn.Module):
    def __init__(
        self,

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
    ):
        super(Transformer_AH, self).__init__()

        self.encoder = Encoder(
            enc_emb_dim=enc_emb_dim,
            encoder_heads=enc_heads,
            encoder_att_layers=enc_layers,
            token_feature_dim=token_feature_dim,
            enc_feed_forward_emb_dim=ff_dim
        )

        self.decoder = MaskedDecoder(
            vocab_size=vocab_size,
            dec_emb_dim=dec_emb_dim,
            decoder_heads=dec_heads,
            decoder_maksed_att_layers=dec_layers,
        )

        self.combiner = Enc_Dec_Combine(
            vocab_size=vocab_size,
            enc_emb_dim = enc_emb_dim, 
            dec_emb_dim = dec_emb_dim, 

            transformer_emb_dim=transformer_emb_dim,
            transformer_heads=comb_heads,
            transfrmer_final_att_layers=comb_layers
        )

    def forward(self, *args, **kwargs):

        mode = kwargs.get("mode", "transformer")    # If mode wasn't passed use the transformer
        enc_in = kwargs.get("enc_in", None)         
        dec_in = kwargs.get("dec_in", None)

        if mode == "encoder":
            #print("Using encoder")
            return self.encoder(enc_in)
        elif mode == "decoder":
            #print("Using decoder")
            return self.decoder(dec_in)
        elif mode == "transformer":
            #print("Using transformer")
            enc_out = self.encoder(enc_in)
            dec_out = self.decoder(dec_in)
            return self.combiner(enc_out, dec_out)
        else:
            raise ValueError(f"Unsupported mode: {mode}")



# ---------------
# --- Encoder ---
# ---------------
class EncoderAttention(torch.nn.Module):

  def __init__(self,enc_emb_dim,encoder_heads):
    super(EncoderAttention, self).__init__()

    # Make sure that embedings dimensions are divisible number of heads
    assert enc_emb_dim % encoder_heads == 0, "embed_dim must be divisible by num_heads"

    self.enc_emb_dim = enc_emb_dim
    self.encoder_heads = encoder_heads
    self.head_dim = enc_emb_dim // encoder_heads

    self.W_Q = torch.nn.Linear(enc_emb_dim, enc_emb_dim)
    self.W_K = torch.nn.Linear(enc_emb_dim, enc_emb_dim)
    self.W_V = torch.nn.Linear(enc_emb_dim, enc_emb_dim)

    # Allows interaction between information from different heads after concatenation
    self.out_proj = torch.nn.Linear(enc_emb_dim, enc_emb_dim)

  def forward(self, inputs):

    B, T, D = inputs.shape # (batch, seq_len, embed_dim)

    # Linear projections
    Q = self.W_Q(inputs)
    K = self.W_K(inputs)
    V = self.W_V(inputs)

    # Reshape to (B, heads, T, head_dim)
    Q = Q.view(B, T, self.encoder_heads, self.head_dim).transpose(1, 2)
    K = K.view(B, T, self.encoder_heads, self.head_dim).transpose(1, 2)
    V = V.view(B, T, self.encoder_heads, self.head_dim).transpose(1, 2)

    # Scaled dot-product attention
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, V)  # (B, H, T, head_dim)

    # Concatenate heads back to (B, T, D)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

    return self.out_proj(attn_output)

class Encoder(torch.nn.Module):
    def __init__(self, token_feature_dim, enc_emb_dim, encoder_heads, encoder_att_layers, enc_feed_forward_emb_dim):
        super(Encoder, self).__init__()

        self.token_feature_dim = token_feature_dim
        self.enc_emb_dim = enc_emb_dim
        self.encoder_heads = encoder_heads
        self.encoder_att_layers = encoder_att_layers
        self.enc_feed_forward_emb_dim = enc_feed_forward_emb_dim

        # Project input features to embedding space
        self.emb = torch.nn.Linear(token_feature_dim, enc_emb_dim)

        # Attention layers
        self.Attns = torch.nn.ModuleList([
            EncoderAttention(enc_emb_dim, encoder_heads) for _ in range(encoder_att_layers)
        ])

        # LayerNorm after attention
        self.attn_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(enc_emb_dim) for _ in range(encoder_att_layers)
        ])

        # Feed-forward sublayers (per layer)
        self.feed1s = torch.nn.ModuleList([
            torch.nn.Linear(enc_emb_dim, enc_feed_forward_emb_dim) for _ in range(encoder_att_layers)
        ])
        self.feed2s = torch.nn.ModuleList([
            torch.nn.Linear(enc_feed_forward_emb_dim, enc_emb_dim) for _ in range(encoder_att_layers)
        ])
        self.ff_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(enc_emb_dim) for _ in range(encoder_att_layers)
        ])

        self.rlu = torch.nn.ReLU()

    def generate_positional_embeddings(self, input_ids):
        if input_ids.dim() == 3:
            batch_size, seq_len, _ = input_ids.size()
        else:
            seq_len = input_ids.size(0)

        device = input_ids.device if input_ids.is_cuda else 'cpu'

        pe = torch.zeros(seq_len, self.enc_emb_dim, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.enc_emb_dim, 2, device=device).float() * (-math.log(10000.0) / self.enc_emb_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        if input_ids.dim() == 3:
            pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, seq_len, dim)

        return pe

    def forward(self, inputs):

        # Initial embedding
        embs = self.emb(inputs.to(dtype=torch.float32))
        pos_embs = self.generate_positional_embeddings(inputs)
        embs = embs + pos_embs

        # Apply each attention + FFN layer
        for Attn, NormAttn, Feed1, Feed2, NormFF in zip(
            self.Attns, self.attn_norms, self.feed1s, self.feed2s, self.ff_norms
        ):
            # Attention + residual + norm
            att_out = Attn(embs)
            embs = NormAttn(embs + att_out)

            # FFN layer
            ff_out = Feed1(embs)
            ff_out = self.rlu(ff_out)
            ff_out = Feed2(ff_out)

            # Residual connection and norm
            embs = NormFF(embs + ff_out)

        return embs




# ---------------
# --- Decoder ---
# ---------------
class DecoderMaskedAttention(torch.nn.Module):
    def __init__(self, dec_emb_dim, decoder_heads):
        super(DecoderMaskedAttention, self).__init__()

        # Ensure dec_emb_dim is divisible by number of heads
        assert dec_emb_dim % decoder_heads == 0, "embed_dim must be divisible by num_heads"

        self.decoder_heads = decoder_heads
        self.head_dim = dec_emb_dim // decoder_heads
        self.dec_emb_dim = dec_emb_dim

        self.W_Q = torch.nn.Linear(dec_emb_dim, dec_emb_dim)
        self.W_K = torch.nn.Linear(dec_emb_dim, dec_emb_dim)
        self.W_V = torch.nn.Linear(dec_emb_dim, dec_emb_dim)

        # Projection back to original embedding size after attention
        self.out_proj = torch.nn.Linear(dec_emb_dim, dec_emb_dim)

    def forward(self, inputs):
        B, T, D = inputs.shape  # Batch, Sequence Length, Embedding Dimension

        # Linear projections (B, T, D)
        Q = self.W_Q(inputs)  
        K = self.W_K(inputs)
        V = self.W_V(inputs)

        # Reshape for multi-head: (B, heads, T, head_dim)
        Q = Q.view(B, T, self.decoder_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.decoder_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.decoder_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)

        # Causal mask to prevent attending to future tokens
        mask = torch.triu(torch.full((T, T), float('-inf'), device=inputs.device), diagonal=1)
        attn_scores = attn_scores + mask 

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, H, T, head_dim)

        # Concatenate heads back (B, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(attn_output)

class MaskedDecoder(torch.nn.Module):
    def __init__(self, vocab_size, dec_emb_dim, decoder_heads, decoder_maksed_att_layers):
        super(MaskedDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.dec_emb_dim = dec_emb_dim
        self.decoder_heads = decoder_heads
        self.decoder_maksed_att_layers = decoder_maksed_att_layers

        # Word embeddings
        self.emb = torch.nn.Embedding(vocab_size, dec_emb_dim)

        # Positional encodings will be added manually
        self.attn_layers = torch.nn.ModuleList([
            DecoderMaskedAttention(dec_emb_dim, decoder_heads) for _ in range(decoder_maksed_att_layers)
        ])
        self.attn_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(dec_emb_dim) for _ in range(decoder_maksed_att_layers)
        ])

        # Feed-forward layers per attention block
        self.feed1s = torch.nn.ModuleList([
            torch.nn.Linear(dec_emb_dim, dec_emb_dim * 4) for _ in range(decoder_maksed_att_layers)
        ])
        self.feed2s = torch.nn.ModuleList([
            torch.nn.Linear(dec_emb_dim * 4, dec_emb_dim) for _ in range(decoder_maksed_att_layers)
        ])
        self.ff_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(dec_emb_dim) for _ in range(decoder_maksed_att_layers)
        ])

        self.relu = torch.nn.ReLU()

    def generate_positional_embeddings(self, input_ids):
        if input_ids.dim() == 2:
            batch_size, seq_len = input_ids.shape
        else:
            seq_len = input_ids.shape[0]
            batch_size = None

        device = input_ids.device if input_ids.is_cuda else 'cpu'

        pe = torch.zeros(seq_len, self.dec_emb_dim, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dec_emb_dim, 2, device=device).float() * (-math.log(10000.0) / self.dec_emb_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        if batch_size is not None:
            pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)

        return pe

    def forward(self, inputs):
        # Get embeddings
        embs = self.emb(inputs)  # (B, T) → (B, T, D)
        pos_embs = self.generate_positional_embeddings(inputs)
        embs = embs + pos_embs  # Add positional information

        # Apply N masked attention + FFN layers
        for Attn, NormAttn, FF1, FF2, NormFF in zip(
            self.attn_layers, self.attn_norms, self.feed1s, self.feed2s, self.ff_norms
        ):
            # Masked self-attention with residual + norm
            attn_out = Attn(embs)
            embs = NormAttn(embs + attn_out)

            # FFN layer
            ff_out = FF1(embs)
            ff_out = self.relu(ff_out)
            ff_out = FF2(ff_out)

            # Residual connection and norm
            embs = NormFF(embs + ff_out)

        return embs




# --------------------------------------------------------
# --- Tranformer - Combine Encoder and Decoder outputs ---
# --------------------------------------------------------
class CombineAttention(torch.nn.Module):
  def __init__(self, enc_emb_dim, dec_emb_dim, transformer_emb_dim, transformer_heads):
    super(CombineAttention, self).__init__()

    assert transformer_emb_dim % transformer_heads == 0, "transformer_emb_dim must be divisible by num_heads"

    self.transformer_heads = transformer_heads
    self.head_dim = transformer_emb_dim // transformer_heads
    self.transformer_emb_dim = transformer_emb_dim

    # Linaer Projections for X-attention
    self.W_Q = torch.nn.Linear(dec_emb_dim, self.transformer_emb_dim)
    self.W_K = torch.nn.Linear(enc_emb_dim, self.transformer_emb_dim)
    self.W_V = torch.nn.Linear(enc_emb_dim, self.transformer_emb_dim)

    # Final projection
    self.out_proj = torch.nn.Linear(transformer_emb_dim, dec_emb_dim)

  def forward(self, out_enc, out_dec):

    B, T_dec, _ = out_dec.shape
    T_enc = out_enc.shape[1]

    Q = self.W_Q(out_dec)
    K = self.W_K(out_enc)
    V = self.W_V(out_enc)

    # Reshape to (B, heads, T, head_dim)
    Q = Q.view(B, T_dec, self.transformer_heads, self.head_dim).transpose(1, 2)
    K = K.view(B, T_enc, self.transformer_heads, self.head_dim).transpose(1, 2)
    V = V.view(B, T_enc, self.transformer_heads, self.head_dim).transpose(1, 2)

    # Scaled dot-product attention
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T_dec, T_enc)
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, V)  # (B, H, T_dec, head_dim)

    # Concatenate heads
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_dec, self.transformer_emb_dim)

    # Final projection to match decoder embedding size
    out = self.out_proj(attn_output)

    return out

class Enc_Dec_Combine(torch.nn.Module):
    def __init__(self, vocab_size, enc_emb_dim, dec_emb_dim, transformer_emb_dim, transformer_heads, transfrmer_final_att_layers): 
        super(Enc_Dec_Combine, self).__init__()

        # Encoder and Decoder embedding dimensions
        self.vocab_size = vocab_size
        self.enc_emb_dim = enc_emb_dim
        self.dec_emb_dim = dec_emb_dim

        # Transformer-Combine variables 
        self.transformer_emb_dim = transformer_emb_dim
        self.transformer_heads = transformer_heads
        self.transfrmer_final_att_layers = transfrmer_final_att_layers 

        # Multi-head attention layers combining encoder & decoder outputs
        self.CombAttns = torch.nn.ModuleList([
            CombineAttention(enc_emb_dim, dec_emb_dim, transformer_emb_dim, transformer_heads)
            for _ in range(transfrmer_final_att_layers)
        ])
        self.combine_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(dec_emb_dim) for _ in range(transfrmer_final_att_layers)
        ])

        # Feed-forward layers
        self.feed1s = torch.nn.ModuleList([
            torch.nn.Linear(dec_emb_dim, transformer_emb_dim) for _ in range(transfrmer_final_att_layers)
        ])
        self.feed2s = torch.nn.ModuleList([
            torch.nn.Linear(transformer_emb_dim, dec_emb_dim) for _ in range(transfrmer_final_att_layers)
        ])
        self.relu = torch.nn.ReLU()

        # Normalisation layers
        self.ff_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(dec_emb_dim) for _ in range(transfrmer_final_att_layers)
        ])

        # Final projection to vocabulary size
        self.vocab = torch.nn.Linear(dec_emb_dim, vocab_size)

    def forward(self, out_enc, out_dec):
        for CombAttn, NormAttn, FF1, FF2, NormFF in zip(
            self.CombAttns, self.combine_norms, self.feed1s, self.feed2s, self.ff_norms
        ):
            # Cross-attention + residual + norm
            attn_out = CombAttn(out_enc, out_dec)
            out_dec = NormAttn(out_dec + attn_out)

            # Feed-forward layer
            ff_out = FF1(out_dec)
            ff_out = self.relu(ff_out)
            ff_out = FF2(ff_out)

            # Residual connection and normalisation
            out_dec = NormFF(out_dec + ff_out)

        # Final projection to vocabulary logits
        logits = self.vocab(out_dec)

        return logits


