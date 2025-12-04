import cv2
import serial
import time

# --- Configuration ---
SERIAL_PORT = 'COM10'
BAUD_RATE = 9600

# --- Camera frame size ---
WIDTH, HEIGHT = 640, 480
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2

# --- Initial Pan/Tilt ---
pan = 0.0
tilt = 0.0
step = 0.1

# Track whether first command was already sent
first_command_sent = False


# -------------------------------
# SERIAL INITIALIZATION
# -------------------------------
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    time.sleep(2)
    print(f"Serial port {SERIAL_PORT} opened successfully.")
except serial.SerialException as e:
    ser = None
    print(f"ERROR: Could not open serial port {SERIAL_PORT}.")
    print(f"Exception: {e}")


# -------------------------------
# WAIT FOR "OK"
# -------------------------------
def wait_for_ok():
    global ser

    if ser is None or not ser.is_open:
        return

    while True:
        try:
            line = ser.readline().decode().strip()
            if line == "OK":
                return
        except:
            pass


# -------------------------------
# SEND SERVO VALUES
# -------------------------------
def send_servo_values(pan_val, tilt_val):
    global ser, first_command_sent

    # WAIT ONLY AFTER FIRST COMMAND
    if first_command_sent:
        wait_for_ok()

    msg = f"{pan_val:.2f} {tilt_val:.2f}\n"

    if ser and ser.is_open:
        try:
            ser.write(msg.encode('utf-8'))
            ser.flush()
        except Exception as e:
            print(f"Serial Write Error: {e}")

    # Mark that the first command has been sent
    first_command_sent = True

    # Debug print
    cam_x = int(CENTER_X + pan_val * CENTER_X)
    cam_y = int(CENTER_Y - tilt_val * CENTER_Y)
    print(f"Sent: Pan={pan_val:.2f}, Tilt={tilt_val:.2f}, Center=({cam_x},{cam_y})")


# -------------------------------
# MANUAL CONTROL
# -------------------------------
def run_manual_control():
    global pan, tilt, step

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- Manual Control Interface ---")
    print("Use WASD to control Pan/Tilt. ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display values
        cv2.putText(frame, f"PAN: {pan:.2f} (A/D)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"TILT: {tilt:.2f} (W/S)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cam_x = int(CENTER_X + pan * CENTER_X)
        cam_y = int(CENTER_Y - tilt * CENTER_Y)
        cv2.drawMarker(frame, (cam_x, cam_y), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"Cam Center: ({cam_x},{cam_y})", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Camera View", frame)
        key = cv2.waitKey(100) & 0xFF

        command_sent = False

        if key == 27:
            break

        if key == ord('a'):
            pan = max(-1.0, pan - step); command_sent = True

        if key == ord('d'):
            pan = min(1.0, pan + step); command_sent = True

        if key == ord('w'):
            tilt = min(1.0, tilt + step); command_sent = True

        if key == ord('s'):
            tilt = max(-1.0, tilt - step); command_sent = True

        if command_sent:
            send_servo_values(pan, tilt)

    cap.release()
    cv2.destroyAllWindows()
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == '__main__':
    run_manual_control()
