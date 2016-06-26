import cv2
import sys
import time
from subprocess import Popen as runAsync

# Global vars
SCALE_FACTOR = 0.5
SEARCH_AREA_FACTOR = 1.5
FRAME_SKIP = 10
FACES_REQUIRED = 20
PROPORTION_LIMIT = 0.3

# Struct for a rectangle
class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def area(self):
        return self.w * self.h

    def midpoint(self):
        return (self.x + self.w/2, self.y + self.h/2)

    def __str__(self):
        return "x:%i, y:%i, w:%i, h:%i" % (self.x, self.y, self.w, self.h)

def alert():
    runAsync(['afplay', 'alert.mp3'])

def cropFrameToRect(frame, rect):
    return frame[rect.y:rect.y+rect.h, rect.x:rect.x+rect.w]

def getLargestFace(faces):
    if len(faces) is 0: 
        return None
    return max(faces, key=lambda x: x.area())

def drawFace(face, frame, color=(255, 0, 0)):
    cv2.rectangle(frame, (face.x, face.y), (face.x+face.w, face.y+face.h), color, 2)

def calculateSearchArea(face, videoRect, factor):
    # TODO: document this shindig
    w = int(factor * face.w)
    h = int(factor * face.h)
    x = max(face.midpoint()[0] - w/2, 0)
    y = max(face.midpoint()[1] - h/2, 0)
    w = min(w, videoRect.w-x)
    h = min(h, videoRect.h-y)
    return Rect(x, y, w, h)

def averageArea(faceList):
    return sum(face.area() for face in faceList) / len(faceList)


# Main class
class PostureTracker:
    def __init__(self):
        # OpenCV stuff
        self.video = cv2.VideoCapture(0)
        self.cascade = cv2.CascadeClassifier("cascade.xml")

        # Set some initial values
        self.currentFrame = None
        self.searchArea = None
        self.faceList = []
        self.alerting = False

        # Used for skipping frames when searchArea is None
        self.counter = 0

        # Get size of video
        _, frame = self.video.read()
        frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        self.videoRect = Rect(0, 0, len(frame[0]), len(frame))

    def detectAndDrawFace(self):

        searchFrame = cropFrameToRect(self.currentFrame, self.searchArea)
        faces = self.cascade.detectMultiScale(cv2.cvtColor(searchFrame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

        # Convert faces to Rect objects and find largest
        faces = [Rect(x, y, w, h) for (x, y, w, h) in faces]
        face = getLargestFace(faces)

        if face:
            # Add back searchArea distance
            face.x += self.searchArea.x
            face.y += self.searchArea.y

            # Draw face and calculate new search area
            drawFace(face, self.currentFrame)

            # Size of full video
            searchArea = calculateSearchArea(face, self.videoRect, SEARCH_AREA_FACTOR)
        else:
            self.searchArea = None

        return face

    def runLoop(self):
        # Capture video and resize
        _, frame = self.video.read()
        self.currentFrame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR) 

        # If no search area, look for face every nth frame
        if self.searchArea is None:
            self.counter += 1
            if self.counter == FRAME_SKIP:
                self.counter = 0
                # Size of full video
                self.searchArea = self.videoRect
        else:
            # Detect and draw!
            face = self.detectAndDrawFace()
            if face:
                self.faceList.append(face)
                if len(self.faceList) > FACES_REQUIRED:
                    self.faceList = self.faceList[1:]
                    avgArea = averageArea(self.faceList)
                    proportion = avgArea / float(self.videoRect.area())
                    if proportion > PROPORTION_LIMIT:
                        if not self.alerting:
                            alert()
                            self.alerting = True
                    else:
                        self.alerting = False
                else:
                    self.alerting = False
            else:
                faceList = []
                self.alerting = False

        # Display the resulting frame
        cv2.imshow('Video', self.currentFrame)

    def start(self):

        # Start run loop
        while True:
            self.runLoop()
            # Quit if q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything is done, release the capture
        self.video.release()
        cv2.destroyAllWindows()

def main():
    tracker = PostureTracker()
    tracker.start()

if __name__ == '__main__':
    main()


