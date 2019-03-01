import numpy as np
import cv2
from PIL import Image
import os
import keras
import pygame
import urllib.request

pygame.init()
pygame.display.set_caption("CIFAR 10")
screen = pygame.display.set_mode((1920,1080),pygame.FULLSCREEN)
pygame.font.init()
myfont = pygame.font.SysFont('Times New Roman', 32)

cap = cv2.VideoCapture(0)
model = keras.models.load_model("CIFAR10v2Chk17.h5")

run = True

print("Entering main loop")

wired = True
try:
    imgResp = urllib.request.urlopen('http://10.190.145.21:8451/shot.jpg')
    wired = False
except:
    wired = True
    

while(run):
    if wired:
        ret, frame = cap.read()
        inputImage = Image.fromarray(frame).resize((32,32)).convert("RGB")
    else:
        imgResp = urllib.request.urlopen('http://10.190.145.21:8451/shot.jpg')
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
        img = cv2.imdecode(imgNp,-1)
        inputImage = Image.fromarray(img).resize((32,32)).convert("RGB")

    
    kerasArray = np.array(inputImage.getdata()).reshape((32,32,3)).reshape((1,32,32,3)) / 255
    compImage = inputImage.resize((320,320))
    
    outputKeras = model.predict(kerasArray)

    mode = compImage.mode
    size = compImage.size
    data = compImage.tobytes()
    
    outImage = pygame.image.fromstring(data, size, mode)
    screen.fill((0,0,0))
    screen.blit(outImage,(0,0))

    array = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
    count = 0
    for val in outputKeras[0]:
        textSurface = myfont.render(array[count], False,(255-int(val/0.003921568627451),int(val/0.003921568627451),0))
        screen.blit(textSurface,(321+(160*count),1010))
        pygame.draw.rect(screen,(255-int(val/0.003921568627451),int(val/0.003921568627451),0),(321+(160*count),1000-int(val*1000),160,int(val*1000)),0)
        count += 1

##    event = pygame.event.wait ()
##    if event.type == pygame.KEYDOWN:
##        if event.key == pygame.K_ESCAPE:
##            run = False

    pygame.display.update()
    
cap.release()
pygame.quit()
