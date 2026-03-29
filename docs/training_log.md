# Training Log

A chronological record of training sessions, bugs discovered, fixes applied, and visual progress.

---

## Session 3 — 2026-03-30

### Summary
Two full 50-epoch training runs completed. Identified and fixed two critical training bugs that were completely preventing the model from learning meaningful inpainting.

### Run 1 — 50 epochs (Original GAN config)

**Config:** `EPOCHS=50`, `BATCH_SIZE=16`, `LR=2e-4`, `lambda_adv=0.1`, `lambda_perceptual=0.05`

| Final Avg G | Final Avg D | Result |
|---|---|---|
| ~1.93 | ~0.606 | 🟫 Flat skin-colored blur |

**Diagnosis:** Classic **blurry mean prediction** failure. `Avg D ≈ 0.606` meant the discriminator was guessing randomly (50/50), providing near-zero gradient signal to the generator. The generator found the "safe" solution: predict the average surrounding skin color for any hole.

**Root cause:** The discriminator had no knowledge of *which region* was inpainted. Seeing the full composed image (where only a 20–50px patch differed), it couldn't distinguish real from fake.

---

### Run 2 — 50 epochs (Conditional PatchGAN + fixes)

**Config:** `EPOCHS=50`, `BATCH_SIZE=16`, `LR=2e-4`, `lambda_adv=0.3`, `lambda_perceptual=0→0.01 (ramped)`

| Epoch | Avg G | Avg D | Notes |
|---|---|---|---|
| 0 | 3.498 | 0.637 | Cold start |
| 1 | 3.013 | 0.495 | D learns fast — conditional signal working ✅ |
| 2 | 2.923 | 0.478 | D still active |
| 5 | 2.601 | 0.529 | Generator catching up |
| 6 | 2.534 | 0.605 | GAN equilibrium begins |
| 16 | 2.371 | 0.596 | Perceptual loss warmup starts |
| 24 | 2.295 | 0.592 | Still slowly declining |
| 49 | 2.196 | 0.602 | Final — G declined 37% from start ✅ |

**Visual result:** Actual color structure and skin-tone patches visible in the hole — no longer a flat blur. Still noisy/inconsistent, but a clear qualitative improvement from Run 1.

**Status:** Model is learning. Further training runs and architectural improvements needed.

---

### Bugs Fixed This Session

#### Bug 1 — VGG Perceptual Loss Weight Caused Green Artifact (Run 1)
- **File:** `train.py`
- **Problem:** `lambda_perceptual = 0.05` scaled a VGG loss that outputs magnitudes of 50–100, making it 10× larger than the pixel loss. The generator was pulled toward matching abstract VGG feature maps before learning any pixel structure, producing a greenish noise blob.
- **Fix:** Progressive warmup schedule — starts at `0.0`, begins ramping linearly at epoch `EPOCHS//3`, reaches `0.01` max at epoch `EPOCHS`. This lets the pixel loss establish structural correctness before perceptual refinement kicks in.

#### Bug 2 — Blind Discriminator Caused Blurry Mean Prediction (Run 1)
- **File:** `src/models/discriminator.py`, `train.py`
- **Problem:** The discriminator received only `comp_img` (3-channel RGB). Since the inpainted hole was a tiny region in the full image, the discriminator couldn't reliably detect fakes and converged to 50/50 random guessing. This provided near-zero gradient to the generator.
- **Fix:** Migrated to a **Conditional PatchGAN**. The discriminator now accepts `cat([image, mask], dim=1)` — 4 channels. It "knows" which region was generated, so it can scrutinize that patch specifically. This is the core pix2pix conditioning technique.
- `lambda_adv` also raised from `0.1` → `0.3` to give the improved adversarial signal more weight.

---

### Next Steps to Improve Quality

1. **More epochs / continued training** — G loss is still declining; more compute will help
2. **Irregular masks** — replace square boxes with brush-stroke masks (see `vision.md`)
3. **Higher resolution** — training at 256×256 instead of 128×128 would allow the model to capture finer texture detail

