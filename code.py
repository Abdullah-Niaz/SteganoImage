from stegano import lsb
from PIL import Image
import math

img_path = "clean_image.jpg"
out_path = "infected_max.jpg"

img = Image.open(img_path)
w, h = img.size

# Max payload in bits using 1 LSB per RGB channel
max_bits = w * h * 3
max_bytes = max_bits // 8

# Use ~95% of capacity to avoid hard failure
payload_size = int(max_bytes * 0.95)

payload = "A" * payload_size

secret = lsb.hide(img_path, payload)
secret.save(out_path)

print(f"Image infected at ~95% capacity")
print(f"Payload size: {payload_size} bytes")
