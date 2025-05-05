from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import base64
import numpy as np
import cv2
import json
import random

app = FastAPI()
app.mount("/photos", StaticFiles(directory="photos"), name="photos")
templates = Jinja2Templates(directory="templates")

net = cv2.dnn.readNetFromONNX("mnist-12.onnx")
classes = ['0','1','2','3','4','5','6','7','8','9']

soru = ""
cevap = 0

with open('sorular.json', 'r') as file:
    sorular = json.load(file)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/eval_canvas")
async def upload_photo(request: Request):
    data = await request.json()
    image_data = data["image"]

    # base64 veriyi ayrıştır
    header, encoded = image_data.split(",", 1)
    binary_data = base64.b64decode(encoded)

    # Fotoğrafı dosyaya kaydet
    with open("photos/islem.png", "wb") as f:
        f.write(binary_data)

    img = cv2.imread("photos/islem.png", cv2.IMREAD_UNCHANGED)
    b,g,r,img2 = cv2.split(img)
    img2 = cv2.bitwise_not(img2)
    img2 = cv2.resize(img2, (28, 28))
    cv2.imwrite("small.png", img2)
    blob = cv2.dnn.blobFromImage(img2, scalefactor=1 / 255, size=(28, 28), mean=[0], swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()

    # En yüksek skor alan sınıfı al
    class_id = np.argmax(outputs)
    confidence = outputs[0][class_id]

    print(f"Tahmin: {classes[class_id]} ({confidence:.2f})")
    if int(classes[class_id]) == cevap:
        return "Doğru cevap! Cevabınız: " + str(classes[class_id])
    else:
        return "Yanlış cevap! Cevabınız: " + str(classes[class_id]) + " Doğru cevap: " + str(cevap)

@app.post("/yeni_soru")
async def soru_yenile():
    global soru, cevap
    soru, cevap = random.choice(list(sorular.items()))
    return soru


