import cv2
import sys
import time
from Rect import Rect
from subprocess import Popen as runAsync

class PostureTracker:
    def __init__(self):
        # OpenCV stuff
        self.video = cv2.VideoCapture(0)
        self.cascade = cv2.CascadeClassifier("cascade.xml")

        # Constants
        self.SEARCH_AREA_FACTOR = 1.5
        self.SCALE_FACTOR = 0.5
        self.FRAME_SKIP = 10
        self.NUM_TO_AVG = 10
        self.PROPORTION_LIMIT = 0.2
        self.GOOD_COLOR = (0, 255, 0)
        self.ALERT_COLOR = (0, 0, 255)

        # Set some initial values
        self.currentFrame = None
        self.searchArea = None
        self.faceList = []
        self.alerting = False
        self.faceColor = self.GOOD_COLOR

        # Used for skipping frames when searchArea is None
        self.counter = 0

        # Get size of video
        _, frame = self.video.read()
        frame = cv2.resize(frame, (0,0), fx=self.SCALE_FACTOR, fy=self.SCALE_FACTOR)
        self.videoRect = Rect(0, 0, len(frame[0]), len(frame))

    def setAlerting(self, doAlert):
        if doAlert:
            if not self.alerting:
                self.alerting = True
                # Play sound
                runAsync(['afplay', 'alert.mp3'])
                self.faceColor = self.ALERT_COLOR
        else:
            if self.alerting:
                self.alerting = False
                self.faceColor = self.GOOD_COLOR

    def cropFrameToRect(self, frame, rect):
        return frame[rect.y:rect.y+rect.h, rect.x:rect.x+rect.w]

    def drawFace(self, face):
        cv2.rectangle(self.currentFrame, (face.x, face.y), (face.x+face.w, face.y+face.h), self.faceColor, 2)

    def calculateSearchArea(self, face, factor):
        # TODO: document this shindig
        w = int(factor * face.w)
        h = int(factor * face.h)
        x = max(face.midpoint()[0] - w/2, 0)
        y = max(face.midpoint()[1] - h/2, 0)
        w = min(w, self.videoRect.w-x)
        h = min(h, self.videoRect.h-y)
        return Rect(x, y, w, h)

    def detectAndDrawFace(self):
        # Get frame within which we'll search
        searchFrame = self.cropFrameToRect(self.currentFrame, self.searchArea)

        # Detect faces
        faces = self.cascade.detectMultiScale(cv2.cvtColor(searchFrame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

        # Convert faces to Rect objects and find largest
        faces = [Rect(x, y, w, h) for (x, y, w, h) in faces]
        face = Rect.largestRect(faces)

        if face:
            # Add back searchArea distance
            face.x += self.searchArea.x
            face.y += self.searchArea.y

            # Draw face and calculate new search area
            self.drawFace(face)
            self.searchArea = self.calculateSearchArea(face, self.SEARCH_AREA_FACTOR)
        else:
            self.searchArea = None

        return face

    def runLoop(self):

        # Capture video and resize
        _, frame = self.video.read()
        self.currentFrame = cv2.resize(frame, (0,0), fx=self.SCALE_FACTOR, fy=self.SCALE_FACTOR) 

        # If no search area, look for face every nth frame
        if self.searchArea is None:
            self.counter += 1
            if self.counter == self.FRAME_SKIP:
                self.counter = 0
                # Set search area to size of full video
                self.searchArea = self.videoRect
        else:
            # Detect and draw!
            face = self.detectAndDrawFace()

            if face:
                # Compile the last n face areas
                self.faceList.append(face)
                if len(self.faceList) > self.NUM_TO_AVG:
                    self.faceList = self.faceList[1:]

                    # Now, decide whether to alert
                    proportion = Rect.avgArea(self.faceList) / self.videoRect.area()
                    if proportion > self.PROPORTION_LIMIT:
                        self.setAlerting(True)
                    else:
                        self.setAlerting(False)

            else:
                faceList = []


        # Display the resulting frame
        cv2.imshow('Video', self.currentFrame)

    def start(self):
        # Start run loop
        print "Tracker started! Press q in window or ^c in terminal to quit."
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
