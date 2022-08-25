import base64
import json
from PIL import Image
import io
# img_b64 = base64.b64decode("")
with open("./image.txt") as f:
    file = json.load(f)

image = file.get("data")
img_b64 = base64.b64decode(image)
image = Image.open(io.BytesIO(img_b64))
image.save("./heatmap2.jpg")
