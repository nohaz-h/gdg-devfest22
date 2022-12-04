import cv2
import os
import numpy as np
from tensorflow import keras

# main function
if __name__ == '__main__':
    vidfile = 'data/videos/for_demo.mp4'
    source = cv2.VideoCapture(vidfile)

    # Load model and labels
    model = keras.models.load_model('./data/model/mymodel.h5')
    with open('./data/model/mymodel.txt', 'r') as l:
        labels = l.read().split('\n')[:-1]

    # Read first image from video feed
    success, image = source.read()
    frame_count = 0
    image_w, image_h, _ = image.shape
    image_min = min(image_w, image_h)
    
    while success:
        # Rotate 180 degrees
        image = cv2.rotate(image, cv2.ROTATE_180)
        frame_count += 1
    
        # Center crop image
        image_crop = image[
            image_w // 2 - image_min // 2:image_w // 2 + image_min // 2,
            image_h // 2 - image_min // 2:image_h // 2 + image_min // 2,
            :
        ]
    
        # Resize to 224 x 224
        image_small = cv2.resize(image_crop, (224, 224))
    
        # Create empty canvas
        canvas = np.ones(shape=(500, 500, 3))

        pred = model.predict(image_small.reshape(1, 224, 224, 3) / 255., verbose=False)

        # Insert video feed into canvas
        canvas[50:50 + 224, 50:50 + 224, :] = image_small / 255.

        for i, (label, prob) in enumerate(zip(labels, pred.flatten())):
            pt1 = (250, 300 + i * 30)
            pt2 = (250 + int(100 * prob), 300 + i * 30)
            canvas = cv2.line(canvas, pt1, pt2, (255, 128, 0), 10)
            outtext = f'{label} ({prob*100:3.0f}%)'
            cv2.putText(canvas, outtext, (50, 300 + i * 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2, cv2.LINE_AA)



        # Show canvas
        cv2.imshow('out', canvas)
        
        # Listen to keypress
        keypress = cv2.waitKey(1000 // 60)
        if keypress == 27: # Esc key
            break
        if keypress == 32: # Spacebar key
            cv2.waitKey(0)
    
        success, image = source.read()
    
    
    print('Done')
    cv2.destroyAllWindows()