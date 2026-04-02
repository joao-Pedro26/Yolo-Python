from ultralytics import YOLO
import cv2
import numpy as np

# def preparar_imagem(caminho_entrada):
#     img = cv2.imread(caminho_entrada)

#     tamanho = 640
#     h, w = img.shape[:2]

#     # Escala mantendo proporção
#     escala = tamanho / max(h, w)
#     nova_largura = int(w * escala)
#     nova_altura = int(h * escala)

#     img_redimensionada = cv2.resize(img, (nova_largura, nova_altura))

#     # Criar imagem preta 640x640
#     nova_img = np.zeros((tamanho, tamanho, 3), dtype=np.uint8)

#     # Centralizar imagem
#     x_offset = (tamanho - nova_largura) // 2
#     y_offset = (tamanho - nova_altura) // 2

#     nova_img[y_offset:y_offset+nova_altura, x_offset:x_offset+nova_largura] = img_redimensionada

#     return nova_img


  

# Caminho da imagem
image_path = "images/train/pessoas2.jpg"

model = YOLO("C:\\Visao-Computacional\\runs\\detect\\train\\weights\\best.pt")

results = model(image_path, imgsz = 640)  # Ajuste o valor de conf conforme necessário

# model = YOLO("yolov8n.pt")  # 'n' = nano (mais leve)

# Faz a detecção
# results = model(image_path, imgsz = 640)

# Pega a imagem com as detecções desenhadas
annotated_image = results[0].plot()

# Mostra a imagem
cv2.imshow("Deteccao YOLO", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salva a imagem (opcional)
cv2.imwrite("resultado.jpg", annotated_image)   

#Contar as pessoas 

contador_pessoas = 0

for box in results[0].boxes:
    classe = int(box.cls[0])
    nome = model.names[classe]

    if nome == "person":
        contador_pessoas += 1

print(f"Total de pessoas detectadas: {contador_pessoas}")
