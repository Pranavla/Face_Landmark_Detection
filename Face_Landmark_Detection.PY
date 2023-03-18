#IMPORTING THE NECESSARY LIBRARIES

import cv2
import itertools
import time as t
import mediapipe as mp

# Initialize the mediapipe face detection class.
mp_face_detection = mp.solutions.face_detection

# Setup the face detection function.
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils


# Initialize the mediapipe face mesh class.
mp_face_mesh = mp.solutions.face_mesh

# Setup the face landmarks function for videos.
face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,  min_detection_confidence=0.5,min_tracking_confidence=0.3)

# Initialize the mediapipe drawing styles class.
mp_drawing_styles = mp.solutions.drawing_styles


# Get the list of indexes of the left and right eye.
LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))

# defining facedectionlandmark fn
def detectFacialLandmarks(frame, face_mesh, display = True):
    # Perform the facial landmarks detection on the input frame, after converting it into RGB format.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Create a copy of the input frame to draw facial landmarks.
    output_frame = frame.copy()

    # Check if facial landmarks in the frame are found.
    if results.multi_face_landmarks:
        # Iterate over the found faces.
        for face_landmarks in results.multi_face_landmarks:
            # Draw the facial landmarks on the output frame with the face mesh tesselation connections using default face mesh tesselation style.
            mp_drawing.draw_landmarks(image=output_frame, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # Draw the facial landmarks on the output frame with the face mesh contours connections using default face mesh contours style.
            mp_drawing.draw_landmarks(image=output_frame, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    return output_frame




#FACE LANDMARK DETECTION ON WEB CAM
camera_video = cv2.VideoCapture(0)

# Set the resolution of the video to 1280x960.
camera_video.set(3,1280)
camera_video.set(4,960)

# Create a named window for resizing purposes.
cv2.namedWindow('Face Landmarks Detection', cv2.WINDOW_NORMAL)

# Initialize a variable to store the time of the previous frame.
time1 = 0

# Iterate until the webcam is accessed successfully.
while True:
    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly then continue to the next iteration to 
    # read the next frame.
    if not ok:
        continue
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Perform Face landmarks detection.
    frame = detectFacialLandmarks(frame, face_mesh_videos)
    
    # Set the time for this frame to the current time.
    prev_time = t.time()
    
    # Check if the difference between the previous and this frame time > 0 to avoid 
    # division by zero.
    if (prev_time - time1) > 0:
    
        # Calculate the number of frames per second.
        frames_per_second = 1.0 / (prev_time - time1)
        
        # Write the calculated number of frames per second on the frame. 
        cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
    # Update the previous frame time to this frame time.
    # As this frame will become previous frame in next iteration.
    time1 = prev_time
    
    # Display the frame.
    cv2.imshow('Face Landmarks Detection', frame)
    
    # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF    
    
    # Check if 'ESC' is pressed and break the loop.
    if(k == 27):
        break

# Release the Face Mesh model and the VideoCapture object, and close the windows.                  
camera_video.release()
cv2.destroyAllWindows()



























