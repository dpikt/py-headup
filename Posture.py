import cv2
import sys
import time
# import subprocess
from subprocess import Popen

# Global vars
SCALE_FACTOR = 0.5
SEARCH_AREA_FACTOR = 1.5
FRAME_SKIP = 10
FACES_REQUIRED = 20
VIDEO = cv2.VideoCapture(0)
CASCADE = cv2.CascadeClassifier("cascade.xml")
PROPORTION_LIMIT = 0.3
ALERTING = False

# Struct for a rectangle
class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def area(self):
        return self.w * self.h

    def distance(self):
        # Based purely on area
        return self.area()

    def midpoint(self):
        return (self.x + self.w/2, self.y + self.h/2)

    def __str__(self):
        return "x:%i, y:%i, w:%i, h:%i" % (self.x, self.y, self.w, self.h)

def alert():
    Popen(['afplay', 'alert.mp3'])

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

def averageArea(faceList):
    return sum(face.area() for face in faceList) / len(faceList)

def detectAndDrawFace(frame, searchArea):

    searchFrame = cropFrameToRect(frame, searchArea)

    faces = CASCADE.detectMultiScale(cv2.cvtColor(searchFrame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    # Convert faces to Rect objects and find largest
    faces = [Rect(x, y, w, h) for (x, y, w, h) in faces]
    face = getLargestFace(faces)

    if face:
        # Add back searchArea distance
        face.x += searchArea.x
        face.y += searchArea.y

        # Draw face and calculate new search area
        drawFace(face, frame)

        # Size of full video
        frameRect = Rect(0, 0, len(frame[0]), len(frame))
        searchArea = calculateSearchArea(face, frameRect, SEARCH_AREA_FACTOR)
    else:
        searchArea = None

    return face, searchArea

def runLoop():
    searchArea = None

    # Used for skipping frames when searchArea is None
    counter = 0
    faceList = []

    # Capture one frame to take measurements
    ret, frame = VIDEO.read()
    frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    frameRect = Rect(0, 0, len(frame[0]), len(frame))

    while True:

        # Capture video and resize
        ret, frame = VIDEO.read()
        frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR) 

        # If no search area, look for face every nth frame
        if searchArea is None:
            counter += 1
            if counter == FRAME_SKIP:
                counter = 0
                # Size of full video
                searchArea = frameRect
        else:
            # Detect and draw!
            face, searchArea = detectAndDrawFace(frame, searchArea)
            if face:
                faceList.append(face)
                if len(faceList) > FACES_REQUIRED:
                    faceList = faceList[1:]
                    avgArea = averageArea(faceList)
                    proportion = avgArea / float(frameRect.area())
                    if proportion > PROPORTION_LIMIT:
                        if not ALERTING:
                            alert()
                            ALERTING = True
                    else:
                        ALERTING = False
                else:
                    ALERTING = False
            else:
                faceList = []
                ALERTING = False

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Quit if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    VIDEO.release()
    cv2.destroyAllWindows()


def main():
    runLoop()


if __name__ == '__main__':
    main()


