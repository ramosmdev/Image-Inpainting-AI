import torch
import random
import numpy as np
from PIL import Image, ImageDraw

def random_mask(img):
    """
    Generate an irregular brush-stroke mask via a random walk.

    Instead of a simple axis-aligned square (easy for the model to exploit),
    we draw a thick polyline that follows a random path across the image.
    This produces organic, blob-like holes that mirror how a user would
    actually brush over an object they want removed.

    Returns a (1, H, W) float32 torch tensor:
        - 1.0 = visible context pixel
        - 0.0 = masked / hidden pixel
    """
    _, h, w = img.shape

    # PIL canvas (white = visible, black = hole)
    pil_mask = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(pil_mask)

    # How many independent strokes to paint
    num_strokes = random.randint(1, 3)

    for _ in range(num_strokes):
        # Brush thickness proportional to image size
        brush_width = random.randint(max(8, w // 32), max(20, w // 12))

        # Random starting point anywhere in the image
        x = random.randint(0, w)
        y = random.randint(0, h)
        points = [(x, y)]

        # Build a random-walk polyline (8–16 steps)
        num_steps = random.randint(8, 16)
        max_step = w // 6   # maximum distance per step

        for _ in range(num_steps):
            x = int(np.clip(x + random.randint(-max_step, max_step), 0, w))
            y = int(np.clip(y + random.randint(-max_step, max_step), 0, h))
            points.append((x, y))

        # Draw the stroke as a thick black line (hole)
        if len(points) >= 2:
            draw.line(points, fill=0, width=brush_width)

    # Convert PIL mask → torch tensor in [0, 1]
    mask_np = np.array(pil_mask, dtype=np.float32) / 255.0   # 1.0=visible, 0.0=hole
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)      # (1, H, W)
    return mask_tensor