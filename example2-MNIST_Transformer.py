import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from transformer_ah import Transformer_AH 

# -----------------------------------
# --- Import image and label data ---
# -----------------------------------

# Define transformation
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

# Load the validation (test) dataset
val_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

#-----------------------------------------------------------------
#--- Ensure reproducibility of results by setting random seeds ---
#-----------------------------------------------------------------

SEED = 42

# Python RNG
random.seed(SEED)

# NumPy RNG
np.random.seed(SEED)

# PyTorch RNG (CPU + GPU)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using multi-GPU

# Ensures deterministic behavior (can slow things down)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




# ---------------------------------------------------------------------
# --- Prepare all encoder input data - both training and validation ---
# ---------------------------------------------------------------------

image_num = 4                           # Specify images in input (1,4,9,16,...)

# Generate 20,000 samples, with each sample having 4 random indices from train_dataset - Used for training
random_array = [random.sample(range(len(train_dataset)), image_num) for _ in range(10000)]

# Generate 500 samples, with each sample having 4 random indices from val_dataset - Used for validation
random_val_array = [random.sample(range(len(val_dataset)), image_num) for _ in range(1000)]

# Store image and label for training
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

# Store image and label for validation
image_val_array = []
label_val_array = []
for nums in random_val_array:
  image_store = []
  label_store = []
  for idx in nums:
    x, y = val_dataset[idx]
    image_store.append(transforms.ToPILImage()(x))
    label_store.append(y)
  image_val_array.append(image_store)
  label_val_array.append(label_store)

# Print results
print("Showing a single image before combining them together  - Visible on Jupyter notebook")
plt.imshow(image_array[0][0], cmap='gray')
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# --- Training data - Combine 4 images together in a square ---
# -------------------------------------------------------------

# Conjugate 4 randomly chosen pictures to a single input
grid_size = int(image_num ** 0.5)       # Number of rows and columns (2x2)
patch_size = 28                         # 28 pixels standard for MNIST images
image_size = grid_size * patch_size     # Pixels in rows and columns (56x56)

# Create array to input in transformer
img_array = []  # Array to store all 4x4 images
for idx in range(len(image_array)):

    # Create blank grayscale image
    img = Image.new('L', (image_size, image_size))

    image_store = image_array[idx]

    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            img.paste(image_store[index], (col * patch_size, row * patch_size, (col + 1) * patch_size, (row + 1) * patch_size))
            index += 1
    img_array.append(img)  # Append the completed image to img_array

# Display the resulting images (for example, the first one in the array)
print("SHowing image used for training  - Visible on Jupyter notebook")
plt.imshow(img_array[0], cmap='gray')
plt.axis('off')
plt.show()

# -----------------------------------------------------------------
# --- Validation data - Combining 4 images together in a square ---
# -----------------------------------------------------------------

img_val_array = []  # Array to store all created images
for idx in range(len(image_val_array)):  # Loop for each image in image_store

    # Create blank grayscale image
    img = Image.new('L', (image_size, image_size))

    image_store = image_val_array[idx]

    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            img.paste(image_store[index], (col * patch_size, row * patch_size, (col + 1) * patch_size, (row + 1) * patch_size))
            index += 1
    img_val_array.append(img)

# Display the resulting images (for example, the first one in the array)
print("Showing image example used for validation - Visible on Jupyter notebook")
plt.imshow(img_val_array[0], cmap='gray')
plt.axis('off')
plt.show()



# Function to extract patches
def extract_patches(img_np, patch_size=14):
    h, w = img_np.shape
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img_np[i:i+patch_size, j:j+patch_size]
            patches.append(patch.flatten())
    return torch.stack([torch.tensor(p, dtype=torch.float32) for p in patches])

overall_grid_size = 16      # Separating above image (2x2) into 16 square grids
smaller_patch_size = 14     # Each square grid has 14x14 pixels

# -----------------------
# Flatten training images
# -----------------------

all_flattened_patches = []  # List to store flattened patches for all images

for img in img_array:

    # Convert the image to a numpy array and reshape into smaller patches
    img_np = np.array(img)
    patches = img_np.reshape(overall_grid_size, smaller_patch_size, smaller_patch_size)

    # Flatten each patch and append to the list
    flattened = extract_patches(img_np, patch_size=14)
    all_flattened_patches.append(flattened)

print(len(all_flattened_patches))
print(all_flattened_patches[0].shape)

# -------------------------
# Flatten validation images
# -------------------------

all_val_flattened_patches = []  # List to store flattened patches for all images

for img in img_val_array:
    # Convert the image to a numpy array and reshape into smaller patches
    img_np = np.array(img)
    patches = img_np.reshape(overall_grid_size, smaller_patch_size, smaller_patch_size)

    # Flatten each patch and append to the list
    flattened_val = extract_patches(img_np, patch_size=14)
    all_val_flattened_patches.append(flattened_val)

print(len(all_val_flattened_patches))
print(all_val_flattened_patches[0].shape)


# Visualise inputs
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(all_flattened_patches[0][i].reshape(14, 14), cmap='gray')
    ax.axis('off')
plt.suptitle("All 16 patches of first training image")
plt.tight_layout()
plt.show()

plt.imshow(all_flattened_patches[0][0].reshape(14, 14), cmap='gray')
plt.title("First patch of first image")
plt.axis('off')
plt.show()

# Display the resulting images (for example, the first one in the array)
plt.imshow(all_flattened_patches[0], cmap='gray')
plt.title("Flattened training image - 16 rows, each row representing a signle patch, 196 columns each column representing the pixels in a patch (14x14)")
plt.axis('off')
plt.show()


# --- ---------------------------------------------
# --- Create label dictionary for decoder input ---
# -------------------------------------------------

# Create dictionary
id2label = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '<s>', 11: '<e>'}
label2id = {v: k for k, v in id2label.items()}
vocab_size = len(id2label)

print("Label Dictionary")
print(id2label)
print(label2id)
print("\nLabel array for first image: ", label_array[0])



# Create dmasked ecoder labels for training
# Two Decoder inputs are needed, one with <s> at beginning and one to 
# hifted right with a <e> at the end
all_label_data = []
for idx, entry in enumerate(label_array):
  label_input = [10] + entry  # Add <s> at beginning of label input
  label_input = torch.tensor(label_input) # Turn label input to tensor

  label_target = entry + [11] # Add <e> at end of label target
  label_target = torch.tensor(label_target) # Turn label target to tensor

  all_label_data.append([label_input, label_target])

# Create decoder labels for validation
# Same as above
all_val_label_data = []
for idx, entry in enumerate(label_val_array):
  label_input = [10] + entry
  label_input = torch.tensor(label_input)

  label_target = entry + [11]
  label_target = torch.tensor(label_target)

  all_val_label_data.append([label_input, label_target])

print("Example of masked decoder token input : ", all_label_data[0])

# --------------------------------------------------------------------
# --- Combine all inputs together to pass in training and val loop ---
# --------------------------------------------------------------------

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

#print("\nFinal input prepared for transformer: \n", full_input[0])









# --------------------------------------
# --- Run training & validation loop ---
# --------------------------------------

# Function to batchify inputs for training loop 

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]

        img_batch = torch.stack([entry[0] for entry in batch])      # (B, 16, 196)
        lbl_in_batch = torch.stack([entry[1] for entry in batch])   # (B, seq_len)
        lbl_tgt_batch = torch.stack([entry[2] for entry in batch])  # (B, seq_len)

        yield img_batch, lbl_in_batch, lbl_tgt_batch

vocab_size = len(label2id)

# --- Define transformer with all of its variables ---
transformer = Transformer_AH(

    # Encoder variables
    enc_emb_dim=96,
    enc_heads=8,
    enc_layers=10,
    ff_dim=256,
    token_feature_dim=196,  # MUST BE EQUAL to last dimension of enc_in 
                            # i.e. enc_in.shape = (batch, seq_len, token_feature_dim)

    # Decorder variables
    vocab_size=vocab_size,
    dec_emb_dim=36,
    dec_heads=4,
    dec_layers=4,

    # Trans-Combine variables
    transformer_emb_dim=64,
    comb_heads=4,
    comb_layers=4,
)



# Loop parameters
lr = 0.00001
epochs = 10
batch_size = 50

optimizer = torch.optim.Adam(transformer.parameters(), lr = lr)

# Define loss function. Add label smoothing
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

# Save results for plotting
train_loss_history = []
val_loss_history = []
val_acc_history = []
entropy_history = []

print("Initiating training loop")

# Training & Validation loop
for epoch in range(epochs):

  # Randomly shuffle the dataset before batching
  random.shuffle(full_input)

  train_loss_epoch = 0

  transformer.train()

  for img_in, lbl_in, lbl_tgt in batchify(full_input, batch_size):

    img_in = img_in.to(torch.float32)
    lbl_in = lbl_in.to(torch.long)
    lbl_tgt = lbl_tgt.to(torch.long)

    optimizer.zero_grad()
    out = transformer(enc_in=img_in, dec_in=lbl_in, mode="transformer") 

    out = out.view(-1, out.shape[-1])
    target_index = lbl_tgt.view(-1)

    #loss = F.cross_entropy(out, target_index)
    loss = loss_fn(out, target_index)

    loss.backward()
    optimizer.step()

    train_loss_epoch += loss.item()
  
  # Validation phase
  val_loss_epoch = 0
  total_val_acc = 0
  with torch.no_grad():  # No gradient computation for validation

    for img_in, lbl_in, lbl_tgt in batchify(full_val_input, batch_size):

      img_in = img_in.to(torch.float32)
      lbl_in = lbl_in.to(torch.long)
      lbl_tgt = lbl_tgt.to(torch.long)

      out = transformer(enc_in=img_in, dec_in=lbl_in, mode="transformer") 

      out = out.view(-1, out.shape[-1])  # Reshape to (batch_size * sequence_length, num_classes)
      target_index = lbl_tgt.view(-1)     # Reshape to (batch_size * sequence_length)

      # Calculate loss
      #val_loss = F.cross_entropy(out, target_index) # Calculate loss using negative loss likelihood
      val_loss = loss_fn(out, target_index)
      val_loss_epoch += val_loss.item()

      preds = torch.argmax(out, dim=-1)  # (B*T,)
      correct = (preds == target_index).float().sum()  # (B*T,) == (B*T,)
      total = preds.numel()
      accuracy = correct / total
      total_val_acc = total_val_acc + accuracy

      probs = F.softmax(out, dim=-1)
      entropy = -(probs * probs.log()).sum(dim=-1)  # shape: (num_tokens,)
      avg_entropy = entropy.mean().item()

  if epoch % 1 == 0:
    print(f"Epoch {epoch}  |  training loss: {train_loss_epoch/len(full_input):.4f}  |  val loss: {val_loss_epoch/len(full_val_input):.4f} | Val acc: {total_val_acc/len(full_val_input):.4f}")

    print("Input tokens:     ", lbl_in[0].tolist())
    print("Target tokens:    ", lbl_tgt[0].tolist())
    print("Predicted tokens: ", preds[:lbl_tgt.size(1)].tolist())
    print(f"Avg token entropy: {avg_entropy:.3f}")
    print("-" * 40)
  

  train_loss_history.append(train_loss_epoch / len(full_input))
  val_loss_history.append(val_loss_epoch / len(full_val_input))
  val_acc_history.append(total_val_acc / len(full_val_input))
  entropy_history.append(avg_entropy)



print("Training loss per epoch: ", train_loss_history)
print("Validation loss per epoch: ", val_loss_history)
print("Validation accuracy per epoch: ", val_acc_history)
print("Validation entropy per epoch: ", entropy_history)
