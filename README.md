Real-Time Pan-Tilt Object Tracker
====================================

This Python project implements a robust, low-latency computer vision system for tracking colored objects and controlling a Pan-Tilt (PT) servo mechanism (like a camera gimbal) via serial communication (e.g., Arduino).

The system uses a **Kalman Filter** for smooth, predictive tracking and incorporates necessary safety features like an **Emergency Face Stop** and **Distance-based Filtering**.

Key Features
--------------

-   **Smooth Control:** Uses a Proportional (P) controller combined with an Output Low-Pass Filter (`SMOOTH`) for fluid, predictable servo movement.

-   **Predictive Tracking:** Employs a **Kalman Filter** to predict the object's next position, minimizing jitter and tracking fast-moving targets reliably.

-   **Dynamic Color Selection:** Allows switching the tracking target color (Red, Yellow, or Blue) in real-time using keyboard inputs.

-   **Distance Filtering:** Filters out noise or objects that are too close/too far by checking the detected contour's radius against `MIN_RADIUS` and `MAX_RADIUS`.

-   **Safety E-Stop:** Integrates a **Haar Cascade** face detector to trigger an emergency stop and return to the home position if a human face is detected for a prolonged period.

-   **Quick Home Return:** Immediately resets the Kalman state and smoothly returns the camera to the center position if the target is lost for a small number of frames (`LOST_REQUIRED`).

🛠️ Setup and Installation
--------------------------

### Prerequisites

1.  **Python 3.x**

2.  **OpenCV (cv2):** For image processing, filtering, and video capture.

3.  **NumPy:** For numerical operations (used heavily by OpenCV and Kalman).

4.  **PySerial:** For communication with the microcontroller.

### Installation Steps

1.  **Clone the Repository (Implied):** Ensure the Python script is saved locally.

2.  **Install Libraries:**

    Bash

    ```
    pip install opencv-python numpy pyserial

    ```

3.  **Hardware Connection:**

    -   Connect your Arduino/Microcontroller with the Pan-Tilt servos attached.

    -   **Crucially:** You must ensure the Arduino is programmed to read serial data in the format `"pan_value,tilt_value\n"` and move the servos accordingly, sending `"OK"` back once the movement is initiated or complete.

⚙️ Configuration
----------------

You must adjust the following constants in the Python script to match your hardware and environment:

| **Constant** | **Description** | **Default** |
| --- | --- | --- |
| `SERIAL_PORT` | COM port for your Arduino (e.g., 'COM10' on Windows, '/dev/ttyUSB0' on Linux) | `'COM10'` |
| `BAUD_RATE` | Must match the baud rate configured on your Arduino | `9600` |
| `PAN_SCALE`, `TILT_SCALE` | **P-Controller Gain.** Controls movement speed. Larger value = slower, smoother response. Tune this! | `700.0` |
| `SMOOTH` | **Output Smoothing Factor.** Low-pass filter applied to the final servo command. Higher value = faster reaction, lower value = smoother output. | `0.08` |
| `MIN_RADIUS`, `MAX_RADIUS` | **Distance Filter.** Defines the size range (in pixels) for valid target objects. | `10`, `100` |
| `LOST_REQUIRED` | Frames without detection before triggering the smooth return home. | `6` |
| `FACE_DETECT_THRESHOLD` | Time (in seconds) a face must be visible to trigger the emergency stop. | `3.0` |

🚀 Running the Tracker
----------------------

1.  Ensure your camera is connected (it uses index `1`: `cv2.VideoCapture(1)`).

2.  Run the script:

    Bash

    ```
    python your_script_name.py

    ```

### Keyboard Controls

| **Key** | **Action** |
| --- | --- |
| **`r`** | Start tracking **RED** objects. |
| **`y`** | Start tracking **YELLOW** objects. |
| **`b`** | Start tracking **BLUE** objects. |
| **`q`** / **`ESC`** | Quit the application and release hardware resources. |

🧠 Technical Deep Dive
----------------------

### Control Mechanism

The system uses a smooth Proportional control loop based on Kalman Filter predictions:

1.  **Error Calculation:** `error_x = pred_x - CENTER_X`

2.  **P-Control:** `target_pan = np.clip(-error_x / PAN_SCALE, -1.0, 1.0)`

3.  **Output Smoothing (Low-Pass Filter):** `pan_cmd = (1 - SMOOTH) * pan_cmd + SMOOTH * target_pan`

This formula ensures the movement is scaled by the error (P-term) and dampened by the `SMOOTH` factor, resulting in the smooth movement observed.

### Kalman Filter Role

The filter uses a 4-state model (`[x, y, v_x, v_y]`) to predict object movement.

-   When the target is lost, the filter continues to predict the object's path until the `LOST_REQUIRED` frame count is hit.

-   Once `LOST_REQUIRED` is reached, the Kalman state is immediately `reset()`, stopping all momentum-based predictions and allowing the home-return logic to take over.

### Distance and Range Filtering

The object tracking is refined by checking the detected contour's radius:

Python

```
# Check 2: Minimum/Maximum Radius (Distance Filter)
if MIN_RADIUS <= radius <= MAX_RADIUS:
    # ... process the contour ...
else:
    # Ignore object (it's too close or too far/small)

```

This prevents the system from being distracted by distant noise or overly large objects that might obscure the camera when they are too close.

**GitHub Repository: https://github.com/Erum330/Realtime_Pan_and_Tilt-Object-Tracking.git **
**Youtube Link: https://youtu.be/4pnOp_KU1B4 **

