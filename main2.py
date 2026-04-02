import numpy as np
import cv2
import tensorflow as tf

# 1. Configurações
MODEL_PATH = "C:\Visao-Computacional\runs\detect\train\weights\best_saved_model\best_float16.tflite"
IMAGE_PATH = "sua_imagem_de_teste.jpg" # <--- MUDE PARA O NOME DA SUA IMAGEM
CONF_THRESHOLD = 0.5  # Sensibilidade (0.5 = 50% de certeza)

# 2. Carregar o Modelo
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'] # Geralmente [1, 640, 640, 3]

# 3. Preparar a Imagem
img_orig = cv2.imread(IMAGE_PATH)
h_orig, w_orig, _ = img_orig.shape
img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (input_shape[2], input_shape[1]))

# Normalização (YOLO espera valores entre 0 e 1)
input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

# 4. Rodar a Inferência
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# 5. Processar Resultados
# O output do YOLOv8 TFLite costuma ser [1, 84, 8400] ou similar
# Precisamos transpor para facilitar a leitura
output = output[0].transpose() 

for prediction in output:
    score = prediction[4:].max() # Pega a maior confiança entre as classes
    if score > CONF_THRESHOLD:
        class_id = np.argmax(prediction[4:])
        
        # Coordenadas (estão em formato centralizado x, y, w, h)
        xc, yc, w, h = prediction[:4]
        
        # Converter para pixels da imagem original
        x1 = int((xc - w/2) * w_orig / input_shape[2])
        y1 = int((yc - h/2) * h_orig / input_shape[1])
        x2 = int((xc + w/2) * w_orig / input_shape[2])
        y2 = int((yc + h/2) * h_orig / input_shape[1])

        # Desenhar no console e na imagem
        print(f"Detectado ID {class_id} com {score:.2f} de confiança")
        cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_orig, f"ID:{class_id} {score:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 6. Mostrar Resultado
cv2.imshow("Resultado TFLite", img_orig)
cv2.waitKey(0)
cv2.destroyAllWindows()