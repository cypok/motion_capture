import os
import imutils
import cv2
import numpy as np
import time
from glob import glob
from collections import deque

# motion detection is comparision between previous_frame && current_frame

FRAMES_TO_PERSIST = 10 # Updates the previous frame in every 10th frame from the loop.
MIN_SIZE_FOR_MOVEMENT = 1200 # higher is the number lesser is motion detection sensitivity. (window size)
MOVEMENT_DETECTED_PERSISTENCE = 20 # no. of frame count down before saving the video.


def millis_to_str(millis):
    # Convert milliseconds to timestamp
    seconds = millis // 1000
    millis %= 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "%d:%02d:%02d.%03d" % (hours, minutes, seconds, millis)


def get_movement_segments(source: str, processed_files_path: str = './recordings'):
    source_basename = os.path.splitext(os.path.basename(source))[0]
    
    cap = cv2.VideoCapture(source) # Then start the webcam
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Init frame variables
    first_frame = None
    next_frame = None

    # Init display font and timeout counters
    font = cv2.FONT_HERSHEY_SIMPLEX
    delay_counter = 0
    movement_persistent_counter = 0

    #fourcc = cv2.VideoWriter_fourcc(*'mjpg')
    #fourcc = cv2.VideoWriter_fourcc(*'h264') # .mp4, doesn't work on macOS for original frame size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # .mp4, works on macOS
    # fourcc = cv2.VideoWriter_fourcc(*'XVID') # .avi

    out = None

    # They are used as pre-frames to captured motion.
    # Their number is limited.
    prev_frames = deque(maxlen=MOVEMENT_DETECTED_PERSISTENCE)

    while True:
        transient_movement_flag = False    
        ret, frame = cap.read()

        millis = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
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

        max_contour = max(cnts, key=cv2.contourArea, default=None)
        if max_contour is not None:
            # Draw max contour rect.
            (x, y, w, h) = cv2.boundingRect(max_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Detect movement only if area is big enough.
            max_contour_area = cv2.contourArea(max_contour)
            if max_contour_area > MIN_SIZE_FOR_MOVEMENT:
                transient_movement_flag = True
        else:
            max_contour_area = 0


        if transient_movement_flag == True:
            movement_persistent_flag = True
            movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

        debug_text = f"counter {str(movement_persistent_counter).zfill(2)} movement area {str(int(max_contour_area)).zfill(5)}"
        if movement_persistent_counter > 0:
            movement_persistent_counter -= 1

        cv2.putText(frame, f"{source_basename} @ {millis_to_str(millis)}", (10,35), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, debug_text, (10,65), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
        frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)
        # cv2.imshow("frame", np.hstack((frame_delta, frame)))
        cv2.imshow("frame", frame)


        # record video -----------
        if movement_persistent_counter>0:
            if not out: # for the very first frame
                height, width, _ = original_frame.shape
                timestamp = millis_to_str(millis).replace(':', '-').replace('.', '-')
                out = cv2.VideoWriter(f'{processed_files_path}/{source_basename}_{timestamp}_{int(time.time())}.mp4',
                                      fourcc, frame_rate, (width, height))
                for f in prev_frames:
                    out.write(f)
            out.write(original_frame)

        # Save each clip in separate video file
        elif movement_persistent_counter == 0 and out:
            out.release()
            out = None

        prev_frames.append(original_frame)

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


def process_pipeline(source_path: str, destination_path: str = './recordings'):
    if os.path.isdir(source_path):
        sources = glob(source_path + "/*.[mM][pP]4") # change to needed
        print(f"start processing: {sources}")
        for source in sources:
            print(source)
            get_movement_segments(source, destination_path)
    elif os.path.isfile(source_path):
        print(f"start processing: {source_path}")
        get_movement_segments(source_path, destination_path)
    else:
        print(f"{source_path} is neither a file nor a directory.")
    exit()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--source_dir", type=str, required=True)
    parser.add_argument("-d", "--destination_dir", type=str, default='./recordings')
    args = parser.parse_args()
    process_pipeline(args.source_dir, args.destination_dir)