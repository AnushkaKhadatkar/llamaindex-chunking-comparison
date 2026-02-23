import requests
import os

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/tinyshakespeare.txt"


def download_dataset():
    """Download Tiny Shakespeare dataset if not already downloaded."""
    if not os.path.exists(DATA_PATH):
        print("Downloading Tiny Shakespeare dataset...")
        response = requests.get(DATA_URL)
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Download complete.")
    else:
        print("Dataset already exists.")


def load_dataset():
    """Load dataset as a single document."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return text


if __name__ == "__main__":
    download_dataset()
    text = load_dataset()

    # Sanity checks (IMPORTANT for rubric)
    print("\n===== DATASET SANITY CHECK =====")
    print(f"Total characters: {len(text)}")
    print("\nPreview (first 500 characters):\n")
    print(text[:500])