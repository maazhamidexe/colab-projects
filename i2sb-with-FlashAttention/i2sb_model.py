import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from flash_attn import flash_attn_qkvpacked_func
from PIL import Image
import os
import time
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ✅ Set PyTorch Memory Optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ✅ Clear Cache Before Running
torch.cuda.empty_cache()

# ✅ Limit Memory Allocation to 80%
torch.cuda.set_per_process_memory_fraction(0.8)

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {root_dir}")
            
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class FlashAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads!"
        
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, seq_len, C = x.shape
        
        # Apply layer norm first
        x = self.norm(x)
        
        # Project to qkv
        qkv = self.qkv(x)
        
        # Reshape qkv for flash attention
        # Flash attention expects shape: (batch_size, seqlen, 3, num_heads, head_dim)
        qkv = qkv.reshape(B, seq_len, 3, self.num_heads, self.head_dim)
        
        # Flash attention forward pass
        attn_output = flash_attn_qkvpacked_func(
            qkv,  # Shape: [B, seq_len, 3, num_heads, head_dim]
            dropout_p=0.0,
            causal=False
        )
        
        # Reshape output back to original dimensions
        # Flash attention output shape: [B, seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape(B, seq_len, C)
        
        # Final projection
        return self.proj(attn_output)


class I2SBBackbone(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=256, num_blocks=6):  # Reduced num_blocks to save memory
        super().__init__()
        
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1)
        )
        
        self.attention_blocks = nn.ModuleList([
            FlashAttentionBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.ffn_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_blocks)
        ])
        
        self.output_conv = nn.Conv2d(hidden_dim, in_channels, 1)
        
    def forward(self, x):
        x = self.init_conv(x)
        B, C, H, W = x.shape  # Ensure correct dimensions
    
        print(f"📏 Before flatten: {x.shape}")  # Debug output
    
        x = x.flatten(2).transpose(1, 2)  # Flatten to (B, H*W, C)
    
        print(f"📏 After flatten: {x.shape}")  # Debug output
    
        for attn, ffn in zip(self.attention_blocks, self.ffn_blocks):
            x = x + attn(x)
            x = x + ffn(x)
    
        x = x.transpose(1, 2).reshape(B, C, H, W)  # Ensure reshape matches original
    
        print(f"📏 After reshape: {x.shape}")  # Debug output
    
        return self.output_conv(x)


def train():
    # ✅ Hyperparameters
    BATCH_SIZE = 8  # Reduced from 512 to prevent OOM
    LEARNING_RATE = 1e-4
    EPOCHS = 15
    MODEL_PATH = "model-flash.pth"
    DATASET_PATH = r"C:\Users\pc\Downloads\patches"
    VAL_SPLIT = 0.1
    ACCUMULATION_STEPS = 4  # ✅ Added gradient accumulation to reduce memory usage

    # ✅ GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageDataset(DATASET_PATH, transform=transform)
    train_size = int((1 - VAL_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        num_workers=0  # ✅ Reduced workers to avoid memory issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=False,
        num_workers=0
    )

    model = I2SBBackbone().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99))
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        torch.cuda.empty_cache()  # ✅ Clears cache each epoch to prevent OOM

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                batch = batch.to(device, non_blocking=True)

                optimizer.zero_grad()

                # ✅ Fix for deprecated autocast warning
                with torch.amp.autocast("cuda"):
                    output = model(batch)
                    loss = criterion(output, batch)

                # ✅ Gradient accumulation (prevents OOM for large batches)
                loss = loss / ACCUMULATION_STEPS
                scaler.scale(loss).backward()

                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

        # ✅ Save model after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(train_loader),
        }, MODEL_PATH)

if __name__ == "__main__":
    train()
