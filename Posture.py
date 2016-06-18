import cv2
import sys
import time

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

# Struct for a rectangle
class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def area(self):
        return self.x * self.y

    def midpoint(self):
        return (self.x + self.w/2, self.y + self.h/2)

    def __str__(self):
        return "x:%i, y:%i, w:%i, h:%i" % (self.x, self.y, self.w, self.h)

def cropFrameToRect(frame, rect):
    return frame[rect.y:rect.y+rect.h, rect.x:rect.x+rect.w]

def getLargestFace(faces):
    if len(faces) is 0: 
        return None
    return max(faces, key=lambda x: x.area())

def drawFace(face, frame, color=(255, 0, 0)):
    cv2.rectangle(frame, (face.x, face.y), (face.x+face.w, face.y+face.h), color, 2)

def calculateSearchArea(face, frameRect, factor):
    # TODO: document this shindig
    w = int(factor * face.w)
    h = int(factor * face.h)
    x = max(face.midpoint()[0] - w/2, 0)
    y = max(face.midpoint()[1] - h/2, 0)
    w = min(w, frameRect.w-x)
    h = min(h, frameRect.h-y)
    return Rect(x, y, w, h)

def main():

    # Global vars
    SCALE_FACTOR = 0.5
    SEARCH_AREA_FACTOR = 1.5
    FRAME_SKIP = 10

    # Starting values
    searchArea = None
    counter = 0

    while True:

        counter += 1

        # Capture video and resize
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR) 
        frameRect = Rect(0, 0, len(frame[0]), len(frame))

        if searchArea is None:
            if counter%FRAME_SKIP is 0:
                searchArea = frameRect
        else:
            # If we have a search area, use it to look for faces
            searchFrame = cropFrameToRect(frame, searchArea)

            faces = faceCascade.detectMultiScale(
                cv2.cvtColor(searchFrame, cv2.COLOR_BGR2GRAY),
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )

            # Convert faces to Rect objects and find largest
            faces = [Rect(x, y, w, h) for (x, y, w, h) in faces]
            face = getLargestFace(faces)

            if face:
                # Add back searchArea distance
                face.x += searchArea.x
                face.y += searchArea.y

                # Draw face and calculate new search area
                drawFace(face, frame)
                searchArea = calculateSearchArea(face, frameRect, SEARCH_AREA_FACTOR)
            else:
                searchArea = None


        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Quit if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


