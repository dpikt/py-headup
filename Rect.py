# Lil class for a rectangle with some util methods

class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def area(self):
        return float(self.w * self.h)

    def midpoint(self):
        return (self.x + self.w/2, self.y + self.h/2)

    def __str__(self):
        return "x:%i, y:%i, w:%i, h:%i" % (self.x, self.y, self.w, self.h)

    # Static

    @staticmethod
    def largestRect(rects):
        if len(rects) is 0: 
            return None
        return max(rects, key=lambda x: x.area())

