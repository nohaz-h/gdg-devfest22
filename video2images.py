import cv2

vidfile = 'data/videos/for_training.mp4'
outdir = 'data/images/train'
source = cv2.VideoCapture(vidfile)

# Read first image from video feed
success, image = source.read()
frame_count = 0
image_w, image_h, _ = image.shape
image_min = min(image_w, image_h)

while success:
    # Rotate 180 degrees
    image = cv2.rotate(image, cv2.ROTATE_180)
    frame_count += 1
    outfile = f'img_{frame_count:04d}.jpg'

    # Center crop image
    image_crop = image[
        image_w // 2 - image_min // 2:image_w // 2 + image_min // 2,
        image_h // 2 - image_min // 2:image_h // 2 + image_min // 2,
        :
    ]

    # Resize to 224 x 224
    image_small = cv2.resize(image_crop, (224, 224))

    # Write image to disk
    cv2.imwrite(outdir + '/' + outfile, image_small)
    print (f'... writing {outfile}')
    cv2.imshow('out', image_small)
    
    # Listen to keypress
    keypress = cv2.waitKey(1000 // 60)
    if keypress == 27: # Esc key
        break
    if keypress == 32: # Spacebar key
        cv2.waitKey(0)

    print (f'Reading frame {frame_count + 1}')
    success, image = source.read()


print('Done')
cv2.destroyAllWindows()