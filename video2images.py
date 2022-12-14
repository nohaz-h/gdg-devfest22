import cv2
import os

# Prepare frame labelling function
def label_frame(frame_no):
    if 1 <= frame_no <= 196:
        return '1984'
    if 218 <= frame_no <= 413:
        return 'donkey'
    if 441 <= frame_no <= 598:
        return 'digital'
    if 613 <= frame_no <= 826:
        return 'rig'
    if 846 <= frame_no <= 1050:
        return 'monument'
    return 'background'

# main function
if __name__ == '__main__':
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
    
        # label current frame
        label = label_frame(frame_count)
        subdir = outdir + '/' + label

        # create directory if not exist
        os.makedirs(outdir + '/' + label) if not os.path.exists(subdir) else None
    
        # Write image to disk
        cv2.imwrite(subdir + '/' + outfile, image_small)
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