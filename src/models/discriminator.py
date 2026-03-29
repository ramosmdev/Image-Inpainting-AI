import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # CONDITIONAL PatchGAN: Takes image (3ch) + mask (1ch) = 4 channels total.
        # By seeing the mask, the discriminator learns "is this hole fill plausible
        # *given the surrounding context*?" rather than blindly classifying the full image.
        # This is the key trick from pix2pix that gives much stronger adversarial signal.
        self.model = nn.Sequential(
            *discriminator_block(4, 64, stride=2, normalize=False),   # 128x128 -> 64x64
            *discriminator_block(64, 128, stride=2),  # 64x64 -> 32x32
            *discriminator_block(128, 256, stride=2), # 32x32 -> 16x16
            *discriminator_block(256, 512, stride=1), # 16x16 -> 15x15
            nn.Conv2d(512, 1, 4, padding=1)           # Output: 1 channel validity map
            # Note: No Sigmoid — BCEWithLogitsLoss expects raw logits!
        )

    def forward(self, img, mask):
        # Concatenate the mask as a 4th conditioning channel
        x = torch.cat([img, mask], dim=1)
        return self.model(x)
