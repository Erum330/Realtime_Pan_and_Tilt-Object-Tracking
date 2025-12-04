import cv2 as cv
import numpy as np

# --- Configuration ---
MIN_CONTOUR_AREA = 500
WIDTH, HEIGHT = 640, 480
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2

# Global variable for the control window name
TRACKBAR_WINDOW = 'HSV Tuning'

# --- Dummy function for trackbar callback (required by OpenCV) ---
def nothing(x):
    pass

# --- 1. Trackbar Initialization ---
def setup_trackbars():
    """Creates the control window and eight trackbars for explicit red HSV tuning."""
    # Resize window to accommodate the two extra Hue sliders
    cv.namedWindow(TRACKBAR_WINDOW)
    cv.resizeWindow(TRACKBAR_WINDOW, 550, 350)
    
    # --- RED RANGE 1 (LOW END) ---
    cv.createTrackbar('H1_min', TRACKBAR_WINDOW, 0, 179, nothing)
    cv.createTrackbar('H1_max', TRACKBAR_WINDOW, 10, 179, nothing)
    
    # --- RED RANGE 2 (HIGH END) ---
    cv.createTrackbar('H2_min', TRACKBAR_WINDOW, 160, 179, nothing)
    cv.createTrackbar('H2_max', TRACKBAR_WINDOW, 179, 179, nothing)
    
    # --- SHARED SATURATION & VALUE ---
    # SATURATION (0 - 255)
    cv.createTrackbar('S_min', TRACKBAR_WINDOW, 150, 255, nothing)
    cv.createTrackbar('S_max', TRACKBAR_WINDOW, 255, 255, nothing)

    # VALUE / BRIGHTNESS (0 - 255)
    cv.createTrackbar('V_min', TRACKBAR_WINDOW, 110, 255, nothing)
    cv.createTrackbar('V_max', TRACKBAR_WINDOW, 255, 255, nothing)

# --- 2. Main Tracking Function ---
def run_tracking_and_vision():
    
    capture = cv.VideoCapture(1) # Changed to 0 for standard laptop webcam
    capture.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not capture.isOpened(): 
        print("Error: Could not open webcam.")
        return
    
    # Initialize the trackbars
    setup_trackbars()

    print("Vision Tracking Active with 4-Slider HSV Tuner. Press 'd' to exit.")

    # Define the kernel used for cleaning
    kernel = np.ones((15,15), np.uint8)

    while True:
        isTrue, frame = capture.read()
        if not isTrue: break

        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hsv_display = hsv_frame.copy() 

        # --- READ TRACKBAR VALUES ---
        # Hue Sliders
        h1_min = cv.getTrackbarPos('H1_min', TRACKBAR_WINDOW)
        h1_max = cv.getTrackbarPos('H1_max', TRACKBAR_WINDOW)
        h2_min = cv.getTrackbarPos('H2_min', TRACKBAR_WINDOW)
        h2_max = cv.getTrackbarPos('H2_max', TRACKBAR_WINDOW)
        
        # S and V (Shared)
        s_min = cv.getTrackbarPos('S_min', TRACKBAR_WINDOW)
        s_max = cv.getTrackbarPos('S_max', TRACKBAR_WINDOW)
        v_min = cv.getTrackbarPos('V_min', TRACKBAR_WINDOW)
        v_max = cv.getTrackbarPos('V_max', TRACKBAR_WINDOW)
        
        # --- MASKING CODE (Explicit Two-Range Red Mask) ---
        
        # Mask 1: Low-end Red (e.g., 0 to 10)
        lower_red_1 = np.array([h1_min, s_min, v_min])
        upper_red_1 = np.array([h1_max, s_max, v_max])
        mask1 = cv.inRange(hsv_frame, lower_red_1, upper_red_1)
        
        # Mask 2: High-end Red (e.g., 160 to 179)
        lower_red_2 = np.array([h2_min, s_min, v_min])
        upper_red_2 = np.array([h2_max, s_max, v_max])
        mask2 = cv.inRange(hsv_frame, lower_red_2, upper_red_2)
        
        # Combine the two masks
        initial_mask = cv.bitwise_or(mask1, mask2)


        # Robust morphology
        cleaned_mask = cv.morphologyEx(initial_mask, cv.MORPH_CLOSE, kernel, iterations=3)
        cleaned_mask = cv.morphologyEx(cleaned_mask, cv.MORPH_OPEN, kernel)

        # --- REMAINDER OF TRACKING LOGIC ---
        
        # Draw the target center (white crosshair)
        CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
        cv.line(frame, (CENTER_X - 15, CENTER_Y), (CENTER_X + 15, CENTER_Y), (255, 255, 255), 2)
        cv.line(frame, (CENTER_X, CENTER_Y - 15), (CENTER_X, CENTER_Y + 15), (255, 255, 255), 2)
        
        contours, _ = cv.findContours(cleaned_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv.contourArea)
            
            if cv.contourArea(largest_contour) > MIN_CONTOUR_AREA:
                
                # Draw the bounding circle
                ((x, y), radius) = cv.minEnclosingCircle(largest_contour)
                center_of_circle = (int(x), int(y))
                visual_radius = int(radius * 0.70) 

                contour_color = (0, 255, 255) 
                cv.circle(frame, center_of_circle, visual_radius, contour_color, 2)
                cv.circle(hsv_display, center_of_circle, visual_radius, contour_color, 2)

                # Centroid Calculation
                M = cv.moments(largest_contour)
                if M["m00"] > 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    
                    # Draw object center (green crosshair)
                    cv.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 2)
                    cv.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 2)
                    
                    # Calculate Tracking Error
                    error_x = center_x - CENTER_X
                    error_y = center_y - CENTER_Y
                    
                    cv.putText(frame, f"X ERROR: {error_x} px", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv.putText(frame, f"Y ERROR: {error_y} px", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv.imshow('1. Original Frame (Tracking Error)', frame)
        cv.imshow('2. HSV Frame (Bounding Circle)', hsv_display)
        cv.imshow('3. Cleaned Binary Mask', cleaned_mask)

        if cv.waitKey(20) & 0xFF == ord("d"):
            break
    print(f"LOWER_RED_1 = np.array([{h1_min}, {s_min}, {v_min}])")
    print(f"UPPER_RED_1 = np.array([{h1_max}, {s_max}, {v_max}])")
    print(f"LOWER_RED_2 = np.array([{h2_min}, {s_min}, {v_min}])")
    print(f"UPPER_RED_2 = np.array([{h2_max}, {s_max}, {v_max}])")

    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    run_tracking_and_vision()