import os
import subprocess
from pathlib import Path

STEHIDE_PATH = r"C:/steghide/steghide.exe"

CLEAN_DIR = r"dataset/clean"
INFECTED_DIR = r"dataset/infected"
PAYLOAD = r"payload.txt"
PASSWORD = "cyber123"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".bmp")

os.makedirs(INFECTED_DIR, exist_ok=True)

for root, _, files in os.walk(CLEAN_DIR):
    for file in files:
        if file.lower().endswith(SUPPORTED_EXTENSIONS):
            clean_path = os.path.join(root, file)

            # Preserve folder structure
            relative_path = os.path.relpath(root, CLEAN_DIR)
            infected_subdir = os.path.join(INFECTED_DIR, relative_path)
            os.makedirs(infected_subdir, exist_ok=True)

            infected_path = os.path.join(infected_subdir, file)

            try:
                subprocess.run(
                    [
                        STEHIDE_PATH,
                        "embed",
                        "-cf", clean_path,
                        "-ef", PAYLOAD,
                        "-sf", infected_path,
                        "-p", PASSWORD,
                        "-f"
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
            except subprocess.CalledProcessError:
                print(f"Failed to infect: {clean_path}")

print("Batch infection completed.")
