import time
import pyglet
from pyglet.window import mouse
from pyglet import shapes
import math
from collections import namedtuple
import os
import random
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

#
# rock paper scissors
#   rectangle is paper, circle is rock, x is scissors
#   code von aufgabe 1 und den rest mit chatgpt viel
#
#
# Set up the Dollar Recognizer
Point = namedtuple('Point', 'X Y')
Rectangle = namedtuple('Rectangle', 'X Y Width Height')
Result = namedtuple('Result', 'Name Score Time')


Numpoints = 64  # Adjust this as per your requirement
Phi = 0.5 * (-1.0 + math.sqrt(5.0))  # Golden Ratio
SquareSize = 250.0
Origin = Point(0, 0)
Diagonal = math.sqrt(SquareSize**2 + SquareSize**2)
HalfDiagonal = 0.5 * Diagonal
AngleRange = math.radians(45.0) / 180  # Convert degrees to radians
AnglePrecision = math.radians(2.0) / 180  # Convert degrees to radians
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
user_score = 0
computer_score = 0
computer_gesture = None
xml_filepath = 'xml_logs2' # 'xml_logs2' hat kleineren datensatz für aufgabe 1

def Resamplepoints(points, n):
    I = PathLength(points) / float(n - 1)
    D = 0.0
    new_points = [points[0]]
    # The idea to use a while loop instead of a for loop was inspired by sabrina
    i = 1
    while i < len(points):
        d = Distance(points[i - 1], points[i])
        if((D + d) >= I):
            qx = points[i-1].X + ((I - D) / d) * (points[i].X - points[i-1].X)
            qy = points[i-1].Y + ((I - D) / d) * (points[i].Y - points[i-1].Y)
            q = Point(qx, qy)
            new_points.append(q)
            points.insert(i, q)
            D = 0
        else:
            D += d
        i += 1
    if(len(new_points) == n - 1):
        new_points.append(Point(points[len(points) - 1].X, points[len(points) - 1].Y))
    return new_points
class DollarRecognizer:
    def __init__(self):
        self.Unistrokes = []

    def AddGesture(self, name, points):
        self.Unistrokes.append(Gesture(name, points))

    def load_unistrokes_from_xml(self):
        for root, subdirs, files in os.walk(xml_filepath):
            if 'ipynb_checkpoint' in root:
                continue
            if len(files) > 0:
                for f in tqdm(files):
                    if '.xml' in f:
                        fname = f.split('.')[0]
                        label = fname[:-2]
                        if label in ["rectangle", "circle", "x"]:
                            xml_root = ET.parse(f'{root}/{f}').getroot()
                            points = []
                            for element in xml_root.findall('Point'):
                                x = element.get('X')
                                y = element.get('Y')
                                points.append(Point(float(x), float(y)))
                            self.AddGesture(label, points)

    def Recognize(self, points):
        t = time.time()
        gesture = Gesture('', points)
        b = float('inf')
        u = -1
        i = 0
        for unistroke in self.Unistrokes:
            d = 0
            d = DistanceAtBestAngle(gesture.points, unistroke.points, -AngleRange, +AngleRange, AnglePrecision)
            if d < b:
                b = d
                u = i
            i += 1
        ts = time.time()
        if u == -1:
            return Result("No match", 0.0, ts-t)
        else:
            return Result(self.Unistrokes[u].name, 1.0 - b / HalfDiagonal, ts-t)


# Helper functions
def Resample(points, n):
    I = PathLength(points) / (n - 1)
    D = 0.0
    newPoints = [points[0]]

    for i in range(1, len(points)):
        d = Distance(points[i - 1], points[i])
        if (D + d) >= I:
            qx = points[i - 1].X + ((I - D) / d) * (points[i].X - points[i - 1].X)
            qy = points[i - 1].Y + ((I - D) / d) * (points[i].Y - points[i - 1].Y)
            q = Point(qx, qy)
            newPoints.append(q)
            points.insert(i, q)
            D = 0.0
        else:
            D += d

    if len(newPoints) == n - 1:
        newPoints.append(Point(points[-1].X, points[-1].Y))

    return newPoints

def IndicativeAngle(points):
    c = Centroid(points)
    return math.atan2(c.Y - points[0].Y, c.X - points[0].X)

def RotateBy(points, radians):
    c = Centroid(points)
    cos = math.cos(radians)
    sin = math.sin(radians)
    newPoints = []

    for i in range(len(points)):
        qx = (points[i].X - c.X) * cos - (points[i].Y - c.Y) * sin + c.X
        qy = (points[i].X - c.X) * sin + (points[i].Y - c.Y) * cos + c.Y
        newPoints.append(Point(qx, qy))

    return newPoints

def ScaleTo(points, size):
    B = BoundingBox(points)
    newPoints = []

    # Avoid division by zero by setting a minimum value
    width = B.Width if B.Width != 0 else 1.0
    height = B.Height if B.Height != 0 else 1.0

    for i in range(len(points)):
        qx = points[i].X * (size / width)
        qy = points[i].Y * (size / height)
        newPoints.append(Point(qx, qy))

    return newPoints

def TranslateTo(points, pt):
    c = Centroid(points)
    newPoints = []

    for i in range(len(points)):
        qx = points[i].X + pt.X - c.X
        qy = points[i].Y + pt.Y - c.Y
        newPoints.append(Point(qx, qy))

    return newPoints

def Vectorize(points):
    sum = 0.0
    vector = []

    for i in range(len(points)):
        vector.append(points[i].X)
        vector.append(points[i].Y)
        sum += points[i].X * points[i].X + points[i].Y * points[i].Y

    magnitude = math.sqrt(sum)
    for i in range(len(vector)):
        vector[i] /= magnitude

    return vector

def PathDistance(pts1, pts2):
    d = 0.0
    for i in range(len(pts1)):
        d += Distance(pts1[i], pts2[i])
    return d / len(pts1)

def PathLength(points):
    d = 0.0
    for i in range(1, len(points)):
        d += Distance(points[i - 1], points[i])
    return d

def Distance(p1, p2):
    dx = p2.X - p1.X
    dy = p2.Y - p1.Y
    return math.sqrt(dx * dx + dy * dy)

def Centroid(points):
    x = 0.0
    y = 0.0
    for i in range(len(points)):
        x += points[i].X
        y += points[i].Y
    x /= len(points)
    y /= len(points)
    return Point(x, y)

def BoundingBox(points):
    minX = float('inf')
    maxX = float('-inf')
    minY = float('inf')
    maxY = float('-inf')

    for i in range(len(points)):
        if points[i].X < minX:
            minX = points[i].X
        if points[i].X > maxX:
            maxX = points[i].X
        if points[i].Y < minY:
            minY = points[i].Y
        if points[i].Y > maxY:
            maxY = points[i].Y

    return Rectangle(minX, minY, maxX - minX, maxY - minY)

def OptimalCosineDistance(v1, v2):
    a = 0.0
    b = 0.0
    for i in range(0, len(v1), 2):
        a += v1[i] * v2[i] + v1[i + 1] * v2[i + 1]
        b += v1[i] * v2[i + 1] - v1[i + 1] * v2[i]
    angle = math.atan(b / a)
    return math.acos(a * math.cos(angle) + b * math.sin(angle))

def DistanceAtBestAngle(points, T, a, b, threshold):
    x1 = Phi * a + (1.0 - Phi) * b
    f1 = DistanceAtAngle(points, T, x1)
    x2 = (1.0 - Phi) * a + Phi * b
    f2 = DistanceAtAngle(points, T, x2)

    while abs(b - a) > threshold:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = Phi * a + (1.0 - Phi) * b
            f1 = DistanceAtAngle(points, T, x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - Phi) * a + Phi * b
            f2 = DistanceAtAngle(points, T, x2)

    return min(f1, f2)

def DistanceAtAngle(points, T, radians):
    newPoints = RotateBy(points, radians)
    return PathDistance(newPoints, T)


class Gesture:
    def __init__(self, name, points):
        self.name = name
        self.points = Resamplepoints(points, Numpoints)
        self.radians = IndicativeAngle(self.points)
        self.points = RotateBy(self.points, - self.radians)
        self.points = ScaleTo(self.points, SquareSize)
        self.points = TranslateTo(self.points, Origin)
        self.vector = Vectorize(self.points)

# Set up Pyglet window

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, "Dollar Recognizer")

batch = pyglet.graphics.Batch()
drawn_lines = []

current_points = []
recognizer = DollarRecognizer()
r = recognizer.load_unistrokes_from_xml()
recognized_gesture = "None"


# Game logic
def get_computer_gesture():
    return random.choice(["rectangle", "circle", "x"])


def determine_winner(user_gesture, computer_gesture):
    global user_score, computer_score
    if user_gesture == computer_gesture:
        return "It's a tie!"
    elif (user_gesture == "rectangle" and computer_gesture == "circle") or \
            (user_gesture == "x" and computer_gesture == "rectangle") or \
            (user_gesture == "circle" and computer_gesture == "x"):
        user_score += 1
        return "You win!"
    else:
        computer_score += 1
        return "Computer wins!"

game_result = "Draw a shape to start!"


@window.event
def on_mouse_press(x, y, button, modifiers):
    global current_points, drawn_lines
    if button == mouse.LEFT:
        current_points = [Point(x, y)]
        drawn_lines = [shapes.Line(x, y, x, y, width=2, batch=batch)]


@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    if buttons & mouse.LEFT:
        current_points.append(Point(x, y))
        last_line = drawn_lines[-1]
        new_line = shapes.Line(last_line.x2, last_line.y2, x, y, width=2, batch=batch)
        drawn_lines.append(new_line)


@window.event
def on_mouse_release(x, y, button, modifiers):
    global recognized_gesture, computer_gesture
    if button == mouse.LEFT:
        current_points.append(Point(x, y))
        result = recognizer.Recognize(current_points)
        recognized_gesture = f"Player gesture: {result.Name} "
        user_gesture = result.Name
        game_result = determine_winner(user_gesture, computer_gesture)
        computer_gesture = get_computer_gesture()
        current_points.clear()


@window.event
def on_draw():
    window.clear()
    batch.draw()
    pyglet.text.Label(
        recognized_gesture,
        font_name='Arial',
        font_size=18,
        x=10, y=window.height - 30,
        anchor_x='left', anchor_y='top'
    ).draw()
    pyglet.text.Label(
        f"game result: {game_result}",
        font_name='Arial',
        font_size=18,
        x=10, y=window.height - 60,
        anchor_x='left', anchor_y='top'
    ).draw()
    pyglet.text.Label(
        f"Computer gesture: {computer_gesture}",
        font_name='Arial',
        font_size=18,
        x=10, y=window.height - 90,
        anchor_x='left', anchor_y='top'
    ).draw()
    pyglet.text.Label(
        f"User Score: {user_score}",
        font_name='Arial',
        font_size=18,
        x=10, y=window.height - 120,
        anchor_x='left', anchor_y='top'
    ).draw()

    pyglet.text.Label(
        f"Computer Score: {computer_score}",
        font_name='Arial',
        font_size=18,
        x=10, y=window.height - 160,
        anchor_x='left', anchor_y='top'
    ).draw()


pyglet.app.run()
