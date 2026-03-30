IMG_SIZE = 256
BATCH_SIZE = 8     # Halved from 16: 256×256 uses ~4× more VRAM than 128×128
LR = 2e-4
EPOCHS = 150       # Tripled from 50: loss was still declining at epoch 49
DATA_PATH = "data/train"