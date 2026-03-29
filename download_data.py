import urllib.request
import zipfile
import os
import sys

# URL for a small, standard dataset: COCO 2017 Validation Set (approx 1 GB, 5000 images)
# It extracts into a single flat directory of images, which perfectly matches your InpaintingDataset.
url = "http://images.cocodataset.org/zips/val2017.zip"
zip_path = "val2017.zip"
data_dir = "data"

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 100 / totalsize
        sys.stdout.write(f"\rDownloading: {percent:.2f}% ({readsofar//(1024*1024)}MB / {totalsize//(1024*1024)}MB)")
        sys.stdout.flush()

if __name__ == "__main__":
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print("Fetching the COCO val2017 dataset (natural images, great for inpainting)...")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path, reporthook)
        print("\nDownload complete!")
    else:
        print(f"\n{zip_path} already exists. Skipping download.")

    print("Extracting images...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # Rename val2017 to train to match config
    if os.path.exists(os.path.join(data_dir, "val2017")):
        if not os.path.exists(os.path.join(data_dir, "train")):
            os.rename(os.path.join(data_dir, "val2017"), os.path.join(data_dir, "train"))
        else:
            print("Warning: data/train already exists, skipping rename.")

    print(f"\nDone! Over 5,000 images are ready in the '{data_dir}/train' folder.")
    print("You can now start training by pointing your dataset to this directory.")
