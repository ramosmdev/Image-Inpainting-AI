import torch
import torch.nn as nn

def conv_block(in_c, out_c):
    """Double conv block with BatchNorm for training stability at high resolution."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

def dilated_conv_block(in_c, out_c, dilation=2):
    """Dilated convolution block to increase receptive field in the bottleneck."""
    # padding = dilation ensures the output spatial dimensions match the input
    pad = dilation
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=pad, dilation=dilation),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=pad, dilation=dilation),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class SelfAttention(nn.Module):
    """
    Self-Attention Layer (Scaled Dot-Product Attention)
    Allows the model to learn long-range global dependencies across the image
    (e.g., matching the symmetry of the left and right eye even if they are far apart).

    Variables represent queries (q), keys (k), and values (v).
    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        # Reduce channel depth by 8 for Q and K to save compute/memory overhead
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        # Learnable scale parameter to smoothly ease the attention into the network
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        N = width * height
        
        # Q: B x C/8 x N -> Transpose to B x N x C/8
        proj_query = self.query_conv(x).view(batch_size, -1, N).permute(0, 2, 1)
        
        # K: B x C/8 x N
        proj_key = self.key_conv(x).view(batch_size, -1, N)
        
        # Attention Matrix: B x N x N
        # (Pixel similarity scores across the whole image)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # V: B x C x N
        proj_value = self.value_conv(x).view(batch_size, -1, N)
        
        # Apply attention scores to V
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        # Skip connection: return the original features + the attended features
        return x + self.gamma * out

class UNet(nn.Module):
    """
    Deeper 4-level U-Net for 256×256 inpainting.

    Encoder path (256→128→64→32→16):
        enc1: 4ch  → 64ch   (image + mask concatenated)
        enc2: 64ch → 128ch
        enc3: 128ch→ 256ch
        enc4: 256ch→ 512ch
        bottleneck: 512ch → 1024ch  at 16×16

    The 4th pooling stage forces the model to build global context across
    1/16th of the image before filling any hole — much stronger than the
    previous 3-level design that bottlenecked at 1/8th.
    """
    def __init__(self):
        super().__init__()

        # --- Encoder ---
        self.enc1 = conv_block(4, 64)       # 256×256 | img(3ch) + mask(1ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)     # 128×128
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(128, 256)    # 64×64
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = conv_block(256, 512)    # 32×32 — new 4th level
        self.pool4 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        # Dilated convolutions here double the receptive field horizontally and vertically
        # without reducing resolution further, allowing the model to "see" better context.
        self.bottleneck = dilated_conv_block(512, 1024, dilation=2)  # 16×16
        
        # Self-Attention block enables dynamic global context
        self.attn = SelfAttention(1024)

        # --- Decoder ---
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)   # 32×32 (skip from enc4)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)    # 64×64 (skip from enc3)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)    # 128×128 (skip from enc2)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)     # 256×256 (skip from enc1)

        # 1×1 conv to map 64 feature channels → 3 RGB channels
        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x, mask):
        # Concatenate mask as a 4th conditioning channel
        x = torch.cat([x, mask], dim=1)

        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck + Self-Attention
        b = self.bottleneck(self.pool4(e4))
        b = self.attn(b)

        # Decode with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Sigmoid bounds output to [0, 1] matching normalized image format
        return torch.sigmoid(self.out(d1))