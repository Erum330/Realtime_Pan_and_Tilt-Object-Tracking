import cv2 as cv

# --- CONFIGURATION ---
WIDTH, HEIGHT = 640, 480
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2

def run_camera_test():
    
    # Camera index 1 for external webcam
    capture = cv.VideoCapture(1) 
    capture.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    if not capture.isOpened(): 
        print("[ERROR] Could not open webcam.")
 
    print(f"[INFO] Camera feed active at {WIDTH}x{HEIGHT} resolution. Press 'ESC' to exit.")



    while True:
        isTrue, frame = capture.read()
        if not isTrue: break

         # Draw a simple crosshair to visually confirm the frame center
        cv.line(frame, (CENTER_X - 15, CENTER_Y), (CENTER_X + 15, CENTER_Y), (0, 255, 0), 2)
        cv.line(frame, (CENTER_X, CENTER_Y - 15), (CENTER_X, CENTER_Y + 15), (0, 255, 0), 2)

        cv.imshow('Camera Test Feed - Press ESC to Exit', frame)


        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
    # Clean up resources
    capture.release()
    cv.destroyAllWindows()
    print("[INFO] Camera test concluded.")

if __name__ == '__main__':
    run_camera_test()        