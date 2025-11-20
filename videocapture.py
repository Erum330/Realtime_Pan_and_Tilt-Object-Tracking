import cv2 as cv

 # Camera index 1 for external webcam
capture = cv.VideoCapture(1)

while True:
        isTrue, frame = capture.read()
        if not isTrue: break

        cv.imshow('Camera Test Feed - Press ESC to Exit', frame)


        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break

capture.release()
cv.destroyAllWindows()