# -*- coding: utf-8 -*-

# Install wandb to save loss calculation. Uncomment all necessary lines that use wandb
#!pip install wandb
#import wandb

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import random
from PIL import Image
import numpy as np
import math


"""**---- Import the image and label data ---**"""

# Define a transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

"""**---- Prepare data - Both training and validation ---**"""

# Conjugate 4 randomly chosen pictures to a single input 
image_num = 4

random_array = [random.sample(range(len(train_dataset)), image_num) for _ in range(5000)]
random_val_array = [random.sample(range(len(train_dataset)), image_num) for _ in range(4000)]

random_idx = random.sample(range(len(train_dataset)), image_num)
random_val_idx = random.sample(range(len(train_dataset)), image_num)

image_array = []
label_array = []
for nums in random_array:
  image_store = []
  label_store = []
  for idx in nums:
    x, y = train_dataset[idx]
    image_store.append(transforms.ToPILImage()(x))
    label_store.append(y)
  image_array.append(image_store)
  label_array.append(label_store)

image_val_array = []
label_val_array = []
for nums in random_val_array:
  image_store = []
  label_store = []
  for idx in nums:
    x, y = train_dataset[idx]
    image_store.append(transforms.ToPILImage()(x))
    label_store.append(y)
  image_val_array.append(image_store)
  label_val_array.append(label_store)

#print(len(image_array))
#print(len(label_array))

#print(len(image_val_array))
#print(len(label_val_array))

#print(random_array)
#print(random_val_array)

# Create a grid of 16 squares
grid_size = int(image_num ** 0.5) #16 sqr = 4
patch_size = 28
image_size = grid_size * patch_size

"""**---- Training data ---**"""
# Assuming `image_store` is a list of images that will be pasted
image_store = [Image.new('L', (patch_size, patch_size)) for _ in range(image_num)]  

# Create array to input in transformer
img_array = []  # Array to store all created images
for idx in range(len(image_array)):  # Loop for each image in image_store
    img = Image.new('L', (image_size, image_size))  # Create a new blank image
    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            img.paste(image_store[index], (col * patch_size, row * patch_size, (col + 1) * patch_size, (row + 1) * patch_size))
            index += 1
    img_array.append(img)  # Append the completed image to img_array

# Display the resulting images (for example, the first one in the array)
#for img in img_array:
#    img.show()
#print(len(img_array))

# Same as above but for one image, created for testing purposes
# img = Image.new('L', (image_size, image_size))
# index = 0
# for row in range(grid_size):
#     for col in range(grid_size):
#         img.paste(image_store[index], (col * patch_size, row * patch_size, (col + 1) * patch_size, (row + 1) * patch_size))
#         index += 1

# img.show()

"""**---- Validation data ---**"""
# Assuming `image_store` is a list of images that will be pasted
image_val_store = [Image.new('L', (patch_size, patch_size)) for _ in range(image_num)]  # Replace with your actual images.

img_val_array = []  # Array to store all created images
for idx in range(len(image_val_array)):  # Loop for each image in image_store
    img = Image.new('L', (image_size, image_size))  # Create a new blank image
    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            img.paste(image_val_store[index], (col * patch_size, row * patch_size, (col + 1) * patch_size, (row + 1) * patch_size))
            index += 1
    img_val_array.append(img)  # Append the completed image to img_array

# Display the resulting images (for example, the first one in the array)
#for img in img_array:
#    img.show()
# print(img_array)
# print(img_val_array)

smaller_patch_size = 14
overall_grid_size = 16

# Assuming img_array contains all the images
all_flattened_patches = []  # List to store flattened patches for all images

for img in img_array:
    # Convert the image to a numpy array and reshape into smaller patches
    img = np.array(img)
    patches = img.reshape(overall_grid_size, smaller_patch_size, smaller_patch_size)

    # Flatten each patch and append to the list
    flattened_patches = torch.tensor([patch.flatten() for patch in patches], dtype=torch.float32)
    all_flattened_patches.append(flattened_patches)

print(len(all_flattened_patches))
print(all_flattened_patches[0].shape)



# Assuming img_array contains all the images
all_val_flattened_patches = []  # List to store flattened patches for all images

for img in img_val_array:
    # Convert the image to a numpy array and reshape into smaller patches
    img = np.array(img)
    patches = img.reshape(overall_grid_size, smaller_patch_size, smaller_patch_size)

    # Flatten each patch and append to the list
    flattened_val_patches = torch.tensor([patch.flatten() for patch in patches], dtype=torch.float32)
    all_val_flattened_patches.append(flattened_patches)

print(len(all_val_flattened_patches))
print(all_val_flattened_patches[0].shape)


#This is the same as above but for one image
# img = np.array(img).reshape(overall_grid_size, smaller_patch_size, smaller_patch_size)
# flattened_patches = torch.tensor(np.array([patch.flatten() for patch in img]), dtype=torch.float32)
# print(flattened_patches.shape)

# patch_pixel_num = flattened_patches.shape[1]
# print(flattened_patches.shape)

"""
**--- Create label dictionary ---**
"""

print(label_store)
id2label = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '<s>', 11: '<e>'}
label2id = {v: k for k, v in id2label.items()}
vocab_size = len(id2label)

print(id2label)
print(label2id)

all_label_data = []
for idx, entry in enumerate(label_array):
  label_input = [10] + entry
  label_input = torch.tensor(label_input)

  label_target = entry + [11]
  label_target = torch.tensor(label_target)

  all_label_data.append([label_input, label_target])


all_val_label_data = []
for idx, entry in enumerate(label_val_array):
  label_input = [10] + entry
  label_input = torch.tensor(label_input)

  label_target = entry + [11]
  label_target = torch.tensor(label_target)

  all_val_label_data.append([label_input, label_target])

print(len(all_label_data))
print(len(all_label_data[0]))
print(all_label_data[0][0].shape)

print(len(all_val_label_data))
print(len(all_val_label_data[0]))
print(all_val_label_data[0][0].shape)

# # Generate Masked decoder input and target
# label_input = [10] + label_store
# label_input = torch.tensor(label_input)

# label_target = label_store + [11]
# label_target = torch.tensor(label_target)

# print(label_input)
# print(label_target)

img_emb_dim_64 = 64
label_emb_dim_32 = 32
dim_128 = 128
print("Encoder input: flattened_patches")
print("Each picture is made of 28x28 pixels. By combining 4 images we have created a 56x56 grid.")
print("The image hwas separated in 16 patches. Each patch has dimensions 14x14 giving it 196 pixels when flattened")
#print(all_flattened_patches_tensor.shape)

print("Decoder inputs (two are needed, one with <s> at beginning and one shifted irght")
print("with a <e> at the end")
#print(all_label_data.shape)

full_input = []
for i in range(len(all_flattened_patches)):
  flattened_patches = all_flattened_patches[i]
  label_input, label_target = all_label_data[i]
  full_input.append([flattened_patches, label_input, label_target])

full_val_input = []
for i in range(len(all_val_flattened_patches)):
  flattened_patches = all_val_flattened_patches[i]
  label_input, label_target = all_val_label_data[i]
  full_val_input.append([flattened_patches, label_input, label_target])

print(len(full_input))
print(len(full_val_input))


"""
**--- Create Encoder ---**
"""

class EncoderAttention(torch.nn.Module):
  def __init__(self,img_emb_dim_64):
    super(EncoderAttention, self).__init__()
    self.W_Q = torch.nn.Linear(img_emb_dim_64, img_emb_dim_64)
    self.W_K = torch.nn.Linear(img_emb_dim_64, img_emb_dim_64)
    self.W_V = torch.nn.Linear(img_emb_dim_64, img_emb_dim_64)

  def forward(self, inputs):
    Q = self.W_Q(inputs)
    K = self.W_K(inputs)
    V = self.W_V(inputs)

    attn = Q @ K.T
    attn = attn/math.sqrt(img_emb_dim_64)
    sftm = F.softmax(attn, dim=1)
    out = sftm @ V
    return out

class FullEncoder(torch.nn.Module):
  def __init__(self, img_emb_dim_64):
    super(FullEncoder, self).__init__()

    self.img_emb_dim_64 = img_emb_dim_64
    self.emb = torch.nn.Linear(196, img_emb_dim_64)
    self.Attns = torch.nn.ModuleList([EncoderAttention(img_emb_dim_64) for _ in range(10)])

    #Feed forward loop
    self.feed1 = torch.nn.Linear(img_emb_dim_64, 3*img_emb_dim_64)
    self.rlu = torch.nn.ReLU()
    self.feed2 = torch.nn.Linear(3*img_emb_dim_64, img_emb_dim_64)

    #Normalisation layer
    self.normalise_LayerNorm = torch.nn.LayerNorm(img_emb_dim_64)

  def generate_positional_embeddings(self, input_ids):
    seq_len = input_ids.size(0)
    pe = torch.zeros(seq_len, self.img_emb_dim_64)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, self.img_emb_dim_64, 2).float() * (-math.log(10000.0) / self.img_emb_dim_64))

    # Apply sine to even indices and cosine to odd indices
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

    return pe

  def forward(self, inputs):

    # Generate embeddings with positional embeddings added
    embs = self.emb(inputs.type(torch.FloatTensor))
    pos_embs = self.generate_positional_embeddings(inputs)
    embs_before_att = embs + pos_embs

    # print("Shape of img_embs, img_pos_embs and embs+img_pos_embs")
    # print(embs.shape)
    # print(pos_embs.shape)
    # print(embs_before_att.shape)
    # print("")

    # Feed embeddings in attention block.
    for Attn in self.Attns: embs_after_att = Attn(embs_before_att)

    # Add residual connection
    embs_after_residual =  embs_after_att + embs_before_att

    # Apply layer normalisation
    embs_after_norm = self.normalise_LayerNorm(embs_after_residual)

    # Carry out feed forward loop
    embs_feed_forward = self.feed1(embs_after_norm)
    embs_feed_forward = self.rlu(embs_feed_forward)
    embs_feed_forward = self.feed2(embs_feed_forward)

    # Add another residual connection
    embs_after_feed_forward = embs_feed_forward + embs_after_residual

    # Apply layer normalisation
    embs_after_feed_forward = self.normalise_LayerNorm(embs_after_feed_forward)

    return embs_after_feed_forward


"""
**--- Create Decoder - Masked attention ---**
"""

class DecoderMaskedAttention(torch.nn.Module):
  def __init__(self, label_emb_dim_32):
    super(DecoderMaskedAttention, self).__init__()
    self.label_emb_dim_32 = label_emb_dim_32
    self.W_Q = torch.nn.Linear(label_emb_dim_32, label_emb_dim_32)
    self.W_K = torch.nn.Linear(label_emb_dim_32, label_emb_dim_32)
    self.W_V = torch.nn.Linear(label_emb_dim_32, label_emb_dim_32)

    #self.normalise_LayerNorm = torch.nn.LayerNorm(label_emb_dim_32)

  def forward(self, inputs):
    Q = self.W_Q(inputs)
    K = self.W_K(inputs)
    V = self.W_V(inputs)

    attn = Q @ K.T
    attn = attn/math.sqrt(self.label_emb_dim_32)

    # Generate masking matrix and add it to attention matrix
    base = torch.full_like(attn, float("-inf"))
    mask = torch.triu(base, diagonal=1)
    masked_attn = attn + mask

    sftm = F.softmax(attn, dim=1)
    out = sftm @ V

    return out

class FullMaskedDecoder(torch.nn.Module):
  def __init__(self, label_emb_dim_32):
    self.label_emb_dim_32 = label_emb_dim_32
    super(FullMaskedDecoder, self).__init__()
    self.emb = torch.nn.Linear(5, label_emb_dim_32)  # Creates initial random embeddings for 4 words/tokens. Each token has 9 embeddings.
    self.MaskedAttns = torch.nn.ModuleList([DecoderMaskedAttention(label_emb_dim_32) for _ in range(10)])

    self.normalise_LayerNorm = torch.nn.LayerNorm(label_emb_dim_32)

  def generate_positional_embeddings(self, input_ids):
    seq_len = input_ids.size(0)
    pe = torch.zeros(seq_len, self.label_emb_dim_32)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, self.label_emb_dim_32, 2).float() * (-math.log(10000.0) / self.label_emb_dim_32))

    # Apply sine to even indices and cosine to odd indices
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

    return pe

  def forward(self, inputs):
    # Take embeddings of input + positional embeddings
    embs = self.emb(inputs.type(torch.FloatTensor))
    pos_embs = self.generate_positional_embeddings(inputs)
    embs_before_att = embs + pos_embs

    # For each attention layer take output embedings and Q values
    for MaskedAttn in self.MaskedAttns: embs_after_att = MaskedAttn(embs_before_att)

    # Add residual connection
    embs_after_residual =  embs_after_att + embs_before_att

    # Apply layer normalisation
    embs_after_norm = self.normalise_LayerNorm(embs_after_residual)

    return embs_after_norm

"""
**--- Combine Encoder and Decoder output ---**
"""

class CombineAttention(torch.nn.Module):
  def __init__(self, label_emb_dim_32, img_emb_dim_64, dim_128, vocab_size):
    super(CombineAttention, self).__init__()

    self.label_emb_dim_32 = label_emb_dim_32
    self.img_emb_dim_64 = img_emb_dim_64
    self.vocab_size = vocab_size
    self.dim_128 = dim_128

    self.W_Q = torch.nn.Linear(label_emb_dim_32, self.dim_128)
    self.W_K = torch.nn.Linear(img_emb_dim_64, self.dim_128)
    self.W_V = torch.nn.Linear(img_emb_dim_64, self.dim_128)

    # This is the final necessary weights used to bring the dimensions of the output down
    self.W_f = torch.nn.Linear(self.dim_128, self.label_emb_dim_32)

  def forward(self, out_enc, out_dec):
    Q = self.W_Q(out_dec)
    K = self.W_K(out_enc)
    V = self.W_V(out_enc)

    # print(Q.shape, K.shape, V.shape, " : Shapes of Q, K and V in combine \n")

    attn = Q @ K.T
    attn = attn/math.sqrt(self.dim_128)

    # print(attn.shape, " :Shape of attn in combine \n")

    sftm = F.softmax(attn, dim=1)
    out = sftm @ V

    # print(out.shape, " : Shape of out before \n")

    # Changing dimensions of outputs to match the out_dec for residual connection
    out = self.W_f(out)

    # print(out.shape, out_dec.shape, " : Shape of out and out_dec  \n")

    out = out + out_dec #Residual connection

    return out

class FullCombine(torch.nn.Module):
  def __init__(self, label_emb_dim_32, img_emb_dim_64, dim_128, vocab_size):
    super(FullCombine, self).__init__()

    self.label_emb_dim_32 = label_emb_dim_32
    self.img_emb_dim_64 = img_emb_dim_64
    self.vocab_size = vocab_size
    self.dim_128 = dim_128

    self.feed1 = torch.nn.Linear(self.label_emb_dim_32, 3*self.dim_128)
    self.rlu = torch.nn.ReLU()
    self.feed2 = torch.nn.Linear(3*self.dim_128, self.label_emb_dim_32)

    self.CombAttns = torch.nn.ModuleList([CombineAttention(label_emb_dim_32, img_emb_dim_64, dim_128, vocab_size) for _ in range(10)])
    self.Encoder = FullEncoder(img_emb_dim_64)
    self.MaskedDecoder = FullMaskedDecoder(label_emb_dim_32)

    # Normalise layer for label/Querries
    self.normalise_LayerNorm_label = torch.nn.LayerNorm(label_emb_dim_32)

    # Normalise layer for label/Querries
    self.normalise_LayerNorm_img = torch.nn.LayerNorm(img_emb_dim_64)

    # Normalise layer to vocabulary
    self.vocab = torch.nn.Linear(label_emb_dim_32, vocab_size)

    #Normalise layer of bigger embedding dimension used
    self.normalise_LayerNorm_128 = torch.nn.LayerNorm(dim_128)

  def forward(self, img_emb_inputs, label_emb_inputs, label_emb_target):

    out_enc = self.Encoder(img_emb_inputs)
    out_dec = self.MaskedDecoder(label_emb_inputs)

    # For each attention layer take output embedings and Q values
    for CombAttn in self.CombAttns: embs_after_att = CombAttn(out_enc, out_dec)

    # print("Embeddings shape after combination")
    # print(embs_after_att.shape)
    # print("")

    # Apply layer normalisation
    embs_after_norm = self.normalise_LayerNorm_label(embs_after_att)

    # print(embs_after_att.shape, " : Embeddings shape after Normalisation \n")

    # Carry out feed forward loop
    embs_feed_forward = self.feed1(embs_after_norm)
    embs_feed_forward = self.rlu(embs_feed_forward)
    embs_feed_forward = self.feed2(embs_feed_forward)

    # print(embs_after_att.shape, " : Embeddings shape after Feed Forward \n")

    # Add another residual connection
    embs_after_feed_forward = embs_feed_forward + embs_after_norm

    # print(embs_after_att.shape, " : Embeddings shape after Feed For and residual conn \n")

    # Apply layer normalisation
    embs_after_final_norm = self.normalise_LayerNorm_label(embs_after_feed_forward)

    # print(embs_after_norm.shape, " : Embeddings shape after final Normalisation \n")

    lgts = self.vocab(embs_after_final_norm) # Carry out a linear layer calculation over the token embeddings
    probs = F.log_softmax(lgts, dim=1)  # Final linear layer - Softmax followed by a logarithm

    return probs

"""
**--- Run training loop & validation loop---**
"""

# Run encoder for images, save Keys and Values embeddings

vocab_size = len(label2id)

transformer = FullCombine(label_emb_dim_32, img_emb_dim_64, dim_128, vocab_size)
#print("Number of parameters: ", torch.numel(transformer.parameters()))

lr = 0.0001
optimizer = torch.optim.Adam(transformer.parameters(), lr = lr)

name = 'CorrectedTransformer_MNIST_500Exs_150E_lr=0.0001'
#wandb.init(project='Transformer', name=name)

for epoch in range(15):

  train_loss_epoch = 0

  transformer.train()
  for input in full_input:

    img_in, lbl_in, lbl_tgt = input
    img_in, lbl_in, lbl_tgt = img_in.type(torch.LongTensor), lbl_in.type(torch.LongTensor), lbl_tgt.type(torch.LongTensor)

    optimizer.zero_grad()
    out = transformer(img_in, lbl_in, lbl_tgt)

    # Reshape out and target_index for CrossEntropyLoss
    out = out.view(-1, out.shape[-1])  # Reshape to (batch_size * sequence_length, num_classes)
    target_index = lbl_tgt.view(-1)     # Reshape to (batch_size * sequence_length)

    # Calculate loss
    loss = F.cross_entropy(out, target_index) # Calculate loss using negative loss likelihood

    loss.backward()
    optimizer.step()

    train_loss_epoch += loss.item()

  # Validation phase
  val_loss_epoch = 0
  with torch.no_grad():  # No gradient computation for validation
    for input in full_val_input:  # Assuming `val` is your validation dataset

      img_in, lbl_in, lbl_tgt = input
      img_in, lbl_in, lbl_tgt = img_in.type(torch.LongTensor), lbl_in.type(torch.LongTensor), lbl_tgt.type(torch.LongTensor)

      out = transformer(img_in, lbl_in, lbl_tgt)

      out = out.view(-1, out.shape[-1])  # Reshape to (batch_size * sequence_length, num_classes)
      target_index = lbl_tgt.view(-1)     # Reshape to (batch_size * sequence_length)

      # Calculate loss
      val_loss = F.cross_entropy(out, target_index) # Calculate loss using negative loss likelihood

      val_loss_epoch += val_loss.item()

      #wandb.log({'loss': train_loss_epoch/len(full_input)})
      #wandb.log({'val_loss' : val_loss_epoch/len(full_val_input)})

  if epoch % 5 == 0:
    print("Epoch ", epoch, " loss: ", train_loss_epoch/len(full_input), "    val loss: ", val_loss_epoch/len(full_val_input))
#wandb.finish()

"""
**--- Evaluate transformer predictions ---**
"""

# Set model to evaluation mode
transformer.eval()

correctOnes = 0
wrongOnes =0

# Disable gradient computation for inference
with torch.no_grad():

  for input in full_val_input:

    img_in, lbl_in, lbl_tgt = input

    # Convert the updated input_tokens list to a tensor
    #inpt = torch.LongTensor(img_in, lbl_in, lbl_tgt) # Add batch dimension if necessary unsqueeze(0)

    # Get model output
    out = transformer(img_in, lbl_in, lbl_tgt)  # Shape: [seq_len, vocab_size]
    result=""
    # Get the last token's prediction (we're predicting the next token)
    for i in range(len(out)):
      next_token_logit = out[i]  # Take the last output in the sequence
      next_token_id = torch.argmax(next_token_logit).item()
      id2label = {v: k for k, v in label2id.items()}
      result += id2label[next_token_id] + " "

    labl=""
    for value in lbl_in:
      labl += id2label[value.item()] + " "

    print("Prediction: ", result)
    print("Label: ", labl)
    print("")


    if result.split(" ")[0:-2] == labl.split(" ")[1:-1]:
      correctOnes +=1
    else:
      wrongOnes +=1

  print(" ")
  print("Correct predictions: ", correctOnes)
  print("Wrong predictions: ", wrongOnes)

