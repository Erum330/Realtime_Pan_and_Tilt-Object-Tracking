import cv2
import time
import numpy as np
import serial

# ================================================================
#                       SERIAL CONFIGURATION
# ================================================================
SERIAL_PORT = 'COM10'    # change accordingly
BAUD_RATE = 9600
SERIAL_OK_TIMEOUT = 1.0  # seconds to wait for "OK" before continuing

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
    time.sleep(2)
    print(f"[INFO] Serial connected to {SERIAL_PORT} @ {BAUD_RATE}.")
except Exception as e:
    ser = None
    print(f"[WARN] Serial not available: {e}. Running without Arduino.")

first_command_sent = False


def wait_for_ok(timeout=SERIAL_OK_TIMEOUT):
    """Wait for Arduino 'OK' with a timeout (so program doesn't hang forever)."""
    if ser is None or not ser.is_open:
        return False

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if line == "OK":
                return True
        except Exception:
            pass
    return False


def send_servo_values(pan_val, tilt_val):
    """
    Send servo pan/tilt to Arduino.
    Format: 'pan,tilt\n' (both floats). Waits for OK on subsequent commands.
    """
    global first_command_sent

    if ser is None or not ser.is_open:
        return

    cmd = f"{pan_val:.3f},{tilt_val:.3f}\n"
    try:
        ser.write(cmd.encode('utf-8'))
        ser.flush()
    except Exception as e:
        print(f"[SERIAL ERROR] write failed: {e}")
        return

    if not first_command_sent:
        first_command_sent = True
        return
    else:
        ok = wait_for_ok()
        if not ok:
            print("[WARN] timed out waiting for OK from Arduino")


# ================================================================
#                       CAMERA + CONSTANTS
# ================================================================
WIDTH, HEIGHT = 640, 480
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2

# === COLOR HSV THRESHOLDS ===
RED_RANGES = [
    (np.array([0, 150, 90]), np.array([10, 255, 255])),
    (np.array([165, 150, 90]), np.array([179, 255, 255]))
]
YELLOW_RANGES = [ 
    (np.array([20, 100, 100]), np.array([35, 255, 255]))
]
BLUE_RANGES = [
    (np.array([100, 100, 100]), np.array([130, 255, 255]))
]
COLOR_MAP = {
    'r': {'name': 'RED', 'ranges': RED_RANGES},
    'y': {'name': 'YELLOW', 'ranges': YELLOW_RANGES},
    'b': {'name': 'BLUE', 'ranges': BLUE_RANGES}
}
tracking_color = 'r' 

MIN_AREA = 500
kernel = np.ones((15, 15), np.uint8)

# Control parameters
PAN_SCALE = 700.0
TILT_SCALE = 700.0
SMOOTH = 0.08             # Output smoothing factor
DEAD_ZONE = 25
STABLE_REQUIRED = 5
LOST_REQUIRED = 6         # frames before returning home (and stopping Kalman influence)
HOME_RETURN_SMOOTH = 0.06
PATH_LENGTH = 30 

# === MIN/MAX RADIUS FOR DISTANCE FILTER ===
MIN_RADIUS = 10 
MAX_RADIUS = 100 

# === E-STOP CONSTANTS ===
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

FACE_DETECT_THRESHOLD = 3.0 # seconds
FRAME_RATE = 30.0 
FRAME_THRESHOLD = int(FACE_DETECT_THRESHOLD * FRAME_RATE)

# state
pan_cmd = 0.0
tilt_cmd = 0.0
kalman_path = [] 

# Kalman filter class (Stable configuration)
class KalmanFilter2D:
    def __init__(self, init_x=CENTER_X, init_y=CENTER_Y):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array([
            [1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 1.]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1., 0., 0., 0.], [0., 1., 0., 0.]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0 
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.kf.statePost = np.array([[np.float32(init_x)], [np.float32(init_y)], 
                                      [0.0], [0.0]], dtype=np.float32)
        self.last_pred_x = init_x
        self.last_pred_y = init_y


    def predict(self):
        p = self.kf.predict()
        px = int(round(float(p[0])))
        py = int(round(float(p[1])))
        self.last_pred_x = px
        self.last_pred_y = py
        return px, py

    def correct(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(meas)

    def reset(self):
        """Resets the Kalman filter state to center."""
        self.kf.statePost = np.array([[np.float32(CENTER_X)], [np.float32(CENTER_Y)], 
                                      [0.0], [0.0]], dtype=np.float32)
        self.last_pred_x = CENTER_X
        self.last_pred_y = CENTER_Y


# instantiate kalman
kf = KalmanFilter2D()

# ================================================================
#                                RUN
# ================================================================
def run_tracking():
    global pan_cmd, tilt_cmd, kf, kalman_path, tracking_color, FRAME_RATE, FRAME_THRESHOLD

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    # Update time-dependent constants based on actual FPS
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 1.0:
             global FRAME_THRESHOLD
             FRAME_RATE = fps
             FRAME_THRESHOLD = int(FACE_DETECT_THRESHOLD * FRAME_RATE)
    except:
        pass
    
    if not cap.isOpened():
        print("[FATAL] Could not open camera. Exiting.")
        return

    stable_frames = 0
    lost_frames = 0
    
    # === EMERGENCY STOP STATE ===
    face_detected_frames = 0
    emergency_stop_active = False

    print("[INFO] Tracking started. Press 'q' or ESC to quit. Press 'r', 'y', or 'b' to change color.")
    kf.reset()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        # --- FACE DETECTION (E-STOP) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            face_detected_frames += 1
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        else:
            face_detected_frames = max(0, face_detected_frames - 1)

        if face_detected_frames >= FRAME_THRESHOLD:
            emergency_stop_active = True
        
        # --- TRACKING LOGIC ---
        if not emergency_stop_active:
            # Masking (Dynamic Color Selection)
            blurred = cv2.GaussianBlur(frame, (7, 7), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            mask = None
            for lower, upper in COLOR_MAP[tracking_color]['ranges']:
                current_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, current_mask) if mask is not None else current_mask
            
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

            # Contouring and Measurement
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            measured = False
            meas_x, meas_y = None, None
            radius = 0 # Initialize radius for UI display
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                # Check 1: Minimum Area
                if area > MIN_AREA:
                    
                    (x, y), radius = cv2.minEnclosingCircle(largest)
                    radius = int(round(radius))
                    
                    # Check 2: Minimum/Maximum Radius (Distance Filter)
                    if MIN_RADIUS <= radius <= MAX_RADIUS:
                        M = cv2.moments(largest)
                        if M.get("m00", 0) != 0:
                            meas_x = int(M["m10"] / M["m00"])
                            meas_y = int(M["m01"] / M["m00"])
                            
                            cv2.circle(frame, (int(x), int(y)), radius, (255, 0, 0), 3)
                            cv2.circle(frame, (meas_x, meas_y), 6, (0, 255, 0), -1)

                            if CENTER_X - WIDTH * 0.45 < meas_x < CENTER_X + WIDTH * 0.45:
                                stable_frames += 1
                            else:
                                stable_frames = 0
                            measured = True
                        else:
                            measured = False
                    else:
                        # Draw filtered out contour in yellow if it fails the radius check
                        cv2.circle(frame, (int(x), int(y)), radius, (0, 255, 255), 2) 
                        measured = False
                else:
                    measured = False

            # Kalman Correction/Prediction Logic
            if measured:
                kf.correct(meas_x, meas_y)
                lost_frames = 0
                # Target for control is the new Kalman prediction
                control_target_x, control_target_y = kf.predict() 
            else:
                lost_frames += 1
                
                # *** MODIFIED LOGIC: Stop Kalman momentum immediately if lost ***
                if lost_frames >= LOST_REQUIRED:
                    # Reset Kalman state/momentum
                    kf.reset()
                    # Clear path history visually
                    kalman_path = []
                    # Control target is now the exact center, triggering smooth return
                    control_target_x, control_target_y = CENTER_X, CENTER_Y
                else:
                    # Brief loss (lost_frames < LOST_REQUIRED): Use the last prediction 
                    # for smooth coasting before reset/home starts
                    control_target_x, control_target_y = kf.predict()


            # Update Kalman path history and Drawing (uses the result of the predict call above)
            pred_x, pred_y = control_target_x, control_target_y
            
            kalman_path.append((pred_x, pred_y))
            if len(kalman_path) > PATH_LENGTH:
                kalman_path.pop(0)
            
            # --- Drawing Kalman Path/Prediction ---
            for i in range(1, len(kalman_path)):
                pt1, pt2 = kalman_path[i-1], kalman_path[i]
                alpha = (i - 1) / PATH_LENGTH
                thickness = int(2 + 2 * alpha)
                color = (int(255 * (1-alpha)), 255, int(255 * alpha)) 
                cv2.line(frame, pt1, pt2, color, thickness)

            cv2.circle(frame, (pred_x, pred_y), 8, (255, 255, 0), 2)
            cv2.putText(frame, "KF", (pred_x + 8, pred_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            # === SERVO CONTROL LOGIC (Home Return) ===
            
            if stable_frames >= STABLE_REQUIRED:
                
                # Calculate Error (Target - Center)
                error_x = control_target_x - CENTER_X
                error_y = control_target_y - CENTER_Y

                # Dead Zone
                if abs(error_x) < DEAD_ZONE: error_x = 0
                if abs(error_y) < DEAD_ZONE: error_y = 0

                # Compute Target Commands (P-Control + Clipping)
                target_pan = np.clip(-error_x / PAN_SCALE, -1.0, 1.0)
                target_tilt = np.clip(-error_y / TILT_SCALE, -1.0, 1.0)

                # Smooth Commands (Low-Pass Filter)
                pan_cmd = (1 - SMOOTH) * pan_cmd + SMOOTH * target_pan
                tilt_cmd = (1 - SMOOTH) * tilt_cmd + SMOOTH * target_tilt

                # Send command
                send_servo_values(pan_cmd, tilt_cmd)
                
                # UI Status
                cv2.putText(frame, f"PAN={pan_cmd:.3f} TILT={tilt_cmd:.3f}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)


            # 2. Home Return Phase (If lost >= LOST_REQUIRED frames)
            elif lost_frames >= LOST_REQUIRED:
                # Home return logic (commands naturally smooth to 0.0 because kf.reset() happened above)
                pan_cmd = (1 - HOME_RETURN_SMOOTH) * pan_cmd + HOME_RETURN_SMOOTH * 0.0
                tilt_cmd = (1 - HOME_RETURN_SMOOTH) * tilt_cmd + HOME_RETURN_SMOOTH * 0.0
                send_servo_values(pan_cmd, tilt_cmd)
                cv2.putText(frame, "Lost. Returning home...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            cv2.imshow("Mask", mask)
            
        else: # Emergency Stop Active
            # Smoothly return to 0,0 command
            pan_cmd = (1 - HOME_RETURN_SMOOTH) * pan_cmd + HOME_RETURN_SMOOTH * 0.0
            tilt_cmd = (1 - HOME_RETURN_SMOOTH) * tilt_cmd + HOME_RETURN_SMOOTH * 0.0
            send_servo_values(pan_cmd, tilt_cmd)
            
            # Display Status
            cv2.putText(frame, "!!! EMERGENCY STOP (FACE DETECTED) !!!", (50, CENTER_Y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Reset state
            kf.reset() # Resets Kalman state/momentum
            kalman_path = []
            stable_frames = 0
            lost_frames = 0
            
            black_mask = np.zeros((HEIGHT, WIDTH, 1), np.uint8)
            cv2.imshow("Mask", black_mask)


        # crosshair & UI
        cv2.drawMarker(frame, (CENTER_X, CENTER_Y), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
        
        status_text = f"Tracking: {COLOR_MAP.get(tracking_color, {}).get('name', 'N/A')} (r/y/b)"
        cv2.putText(frame, status_text, (10, HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        info_text = f"Measured={'Y' if measured else 'N'} Lost={lost_frames}/{LOST_REQUIRED} E-Stop={face_detected_frames}/{FRAME_THRESHOLD} Radius={radius if measured else '--'}"
        cv2.putText(frame, info_text, (10, HEIGHT-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        
        # KEYPRESS LOGIC (r, y, b, q)
        if key == ord('q') or key == 27:
            break
        elif key in [ord('r'), ord('y'), ord('b')]:
            if not emergency_stop_active:
                new_color = chr(key)
                if new_color != tracking_color:
                    tracking_color = new_color
                    kf.reset()
                    kalman_path = []
                    stable_frames = 0
                    lost_frames = 0
                    print(f"[INFO] Tracking color changed to {COLOR_MAP[tracking_color]['name']}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracking()