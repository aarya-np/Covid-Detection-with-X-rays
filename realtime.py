import cv2
import tensorflow as tf
import numpy as np
model=tf.keras.models.load_model('model.h5')
def get_predictions(frame):
    predictions=model.predict(frame)
    index=np.argmax(predictions)
    label=['Covid','Normal','Viral Pneumonia']
    return label[index]
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print('camera not working')
else:
    while(True):
        ret,frame=cap.read()
        scaled=cv2.resize(frame,(300,300))
        normalized=scaled/255.0
        final=np.expand_dims(normalized,axis=0)
        label=get_predictions(final)
        cv2.putText(frame,f'Predictions :{label}',(10,30),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Camera Feed',frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
