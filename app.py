from fastapi import FastAPI, File, UploadFile
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI()

# Load the model
torch.set_float32_matmul_precision("high")
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cuda")

# Image transformations
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def process(image):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to("cuda")
    
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

@app.post("/remove-bg/")
async def remove_bg(file: UploadFile = File(...)):
    try:
        # Load image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        
        # Process image
        output = process(image)

        # Convert to bytes
        img_io = io.BytesIO()
        output.save(img_io, format="PNG")
        img_io.seek(0)

        return {"message": "Success", "data": img_io.getvalue()}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import os
    import uvicorn
    PORT = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
