import imutils
import cv2
import numpy as np
import time
from glob import glob

# motion detection is comparision between previous_frame && current_frame

FRAMES_TO_PERSIST = 10 # Updates the previous frame in every 10th frame from the loop.
MIN_SIZE_FOR_MOVEMENT = 200 # higher is the number lesser is motion detection sensitivity. (window size)
MOVEMENT_DETECTED_PERSISTENCE = 100 # no. of frame count down before saving the video.


def get_movement_segments(source: str, processed_files_path: str = './recordings'):
    
    cap = cv2.VideoCapture(source) # Then start the webcam
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Init frame variables
    first_frame = None
    next_frame = None

    # Init display font and timeout counters
    font = cv2.FONT_HERSHEY_SIMPLEX
    delay_counter = 0
    movement_persistent_counter = 0

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fourcc = cv2.VideoWriter_fourcc(*'h264') # .mp4

    # fourcc = cv2.VideoWriter_fourcc(*'XVID') # .avi
    out = None

    while True:
        transient_movement_flag = False    
        ret, frame = cap.read()
        text = "Unoccupied"
        
        if not ret:
            print("CAPTURE ENDED")
            cv2.destroyAllWindows()

            if out != None:
                out.release()

            cap.release()
            return
            
        
        original_frame = frame.copy()

        frame = imutils.resize(frame, width = 750)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None: first_frame = gray    

        delay_counter += 1

        if delay_counter > FRAMES_TO_PERSIST:
            delay_counter = 0
            first_frame = next_frame

            
        next_frame = gray

        frame_delta = cv2.absdiff(first_frame, next_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations = 2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
                transient_movement_flag = True
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if transient_movement_flag == True:
            movement_persistent_flag = True
            movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

        if movement_persistent_counter > 0:
            text = "Movement Detected " + str(movement_persistent_counter)
            movement_persistent_counter -= 1
        else:
            text = "No Movement Detected"

        # cv2.putText(frame, str(text), (10,35), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
        frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)
        # cv2.imshow("frame", np.hstack((frame_delta, frame)))
        cv2.imshow("frame", frame)


        # record video -----------
        if movement_persistent_counter>0 and not out: # for the very first frame
            height, width, _ = frame.shape
            out = cv2.VideoWriter(f'{processed_files_path}/{int(time.time())}_video.mp4',fourcc, frame_rate ,(width, height))
            out.write(frame)

        if movement_persistent_counter>0 and out:
            out.write(frame)

        # Save each clip in separate video file
        if movement_persistent_counter == 0 and out:
            out.release()
            out = None
        

        ch = cv2.waitKey(1)
        if ch & 0xFF == ord('q'):
            if out is not None: # saving & exiting on press of q
                out.release()
                out = None
            break

    # Cleanup when closed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if out != None:
        out.release()

    cap.release()
    return


def process_pipeline(source_path: str, destination_path: str = './recordings', mode: str ='dir'):
    if mode == 'dir':
        print("start processing")
        sources = glob(source_path + "*.[mM][pP]4") # change to needed 
        print(sources)
        for source in sources:
            print(source)
            get_movement_segments(source, destination_path)
    elif mode=='file': 
        get_movement_segments(source_path, destination_path)
    exit()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--source_dir", type=str)
    parser.add_argument("-d", "--destination_dir", type=str, default='./recordings')
    parser.add_argument("-m", "--mode", type=str, default="dir", choices=['dir', 'file'])
    args = parser.parse_args()
    process_pipeline(args.source_dir, args.destination_dir, args.mode)