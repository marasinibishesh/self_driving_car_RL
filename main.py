import numpy as np
from random import randint
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.lang import Builder
from PIL import Image as PILImage
from agent import Dqn
import time
import os

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Global variables
last_x = 0
last_y = 0
n_points = 0
length = 0
states_i = 5  # 3 sensors + orientation + distance
actions_i = 3  # Straight, left, right
gamma_i = 0.9
model = Dqn(states_i, actions_i, gamma_i)
action2rotation = [0, 20, -20]
scores = []
sand = None
max_distance = 1  # Will be calculated later

tracks = [os.path.join('tracks', f'track{i}.png') for i in range(1, 5)]
overlays = [os.path.join('tracks', f'track{i}-overlay.png') for i in range(1, 5)]
current_track_index = 0

def load_sand(track_path, size):
    img = PILImage.open(track_path).convert('L')
    img = img.resize(size, PILImage.Resampling.LANCZOS)
    img_array = np.array(img)
    return (img_array > 128).astype(int)

class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor_x1 = NumericProperty(0)
    sensor_y1 = NumericProperty(0)
    sensor_1 = ReferenceListProperty(sensor_x1, sensor_y1)
    sensor_x2 = NumericProperty(0)
    sensor_y2 = NumericProperty(0)
    sensor_2 = ReferenceListProperty(sensor_x2, sensor_y2)
    sensor_x3 = NumericProperty(0)
    sensor_y3 = NumericProperty(0)
    sensor_3 = ReferenceListProperty(sensor_x3, sensor_y3)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor_1 = Vector(45, 0).rotate(self.angle) + self.pos  # Increased sensor range
        self.sensor_2 = Vector(45, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor_3 = Vector(45, 0).rotate((self.angle - 30) % 360) + self.pos
        
        # Calculate sensor values with proper bounds checking
        self.calculate_sensor_value(1, self.sensor_x1, self.sensor_y1)
        self.calculate_sensor_value(2, self.sensor_x2, self.sensor_y2)
        self.calculate_sensor_value(3, self.sensor_x3, self.sensor_y3)

    def calculate_sensor_value(self, sensor_num, x, y):
        try:
            y_int = int(y)
            x_int = int(x)
            h, w = sand.shape[0], sand.shape[1]
            
            # Increased sensor area to 7x7 box
            y_start = max(0, y_int - 3)
            y_end = min(h, y_int + 4)
            x_start = max(0, x_int - 3)
            x_end = min(w, x_int + 4)
            
            # Calculate sand density in sensor area
            sand_area = sand[y_start:y_end, x_start:x_end]
            sand_count = np.sum(sand_area)
            total_pixels = (y_end - y_start) * (x_end - x_start)
            
            if total_pixels == 0:
                sand_density = 1.0
            else:
                sand_density = sand_count / total_pixels
            
            # Set sensor value (convert to float)
            if sensor_num == 1:
                self.signal1 = float(sand_density)
            elif sensor_num == 2:
                self.signal2 = float(sand_density)
            elif sensor_num == 3:
                self.signal3 = float(sand_density)
                
        except IndexError:
            if sensor_num == 1:
                self.signal1 = 1.0
            elif sensor_num == 2:
                self.signal2 = 1.0
            elif sensor_num == 3:
                self.signal3 = 1.0

class Ball1(Widget):
    pass

class Ball2(Widget):
    pass

class Ball3(Widget):
    pass

class Game(Widget):
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    x_destination = NumericProperty(20)
    y_home = NumericProperty(0)
    destination_rewarded = False
    max_distance = 1
    last_reward = 0
    stuck_timer = 0
    last_save_time = 0
    save_interval = 300  # 5 minutes in seconds
    scores = []
    first_update = True

    def __init__(self, **kwargs):
        super(Game, self).__init__(**kwargs)
        self.track_image = Image(source=tracks[0], allow_stretch=True, keep_ratio=False)
        self.car = Car()
        self.overlay_image = Image(source=overlays[0], allow_stretch=True, keep_ratio=False)
        self.add_widget(self.track_image)
        self.add_widget(self.car)
        self.add_widget(self.overlay_image)
        self.ball1 = Ball1()
        self.ball2 = Ball2()
        self.ball3 = Ball3()
        self.add_widget(self.ball1)
        self.add_widget(self.ball2)
        self.add_widget(self.ball3)
        self.bind(size=self.on_size, pos=self.on_size)
        global sand
        sand = load_sand(tracks[0], (int(self.width), int(self.height)))
        self.y_home = self.height - 20
        self.max_distance = np.sqrt(self.width**2 + self.height**2)
        self.first_update = True
        self.destination_rewarded = False
        self.last_save_time = time.time()

    def on_size(self, *args):
        global sand
        self.track_image.size = self.size
        self.track_image.pos = self.pos
        self.overlay_image.size = self.size
        self.overlay_image.pos = self.pos
        sand = load_sand(tracks[current_track_index], (int(self.width), int(self.height)))
        self.y_home = self.height - 20
        self.max_distance = np.sqrt(self.width**2 + self.height**2)
        self.first_update = True
        self.destination_rewarded = False

    def switch_track(self):
        global current_track_index, sand
        current_track_index = (current_track_index + 1) % len(tracks)
        self.track_image.source = tracks[current_track_index]
        self.overlay_image.source = overlays[current_track_index]
        sand = load_sand(tracks[current_track_index], (int(self.width), int(self.height)))
        self.x_destination = 20
        self.y_home = self.height - 20
        self.max_distance = np.sqrt(self.width**2 + self.height**2)
        self.serve_car()
        self.first_update = True
        self.last_reward = 0
        self.scores = []
        self.destination_rewarded = False

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)
        self.car.angle = 0
        self.destination_rewarded = False

    def update(self, dt):
        xx = self.x_destination - self.car.x
        yy = self.y_home - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.0
        distance = np.sqrt((self.car.x - self.x_destination)**2 + (self.car.y - self.y_home)**2)
        normalized_distance = distance / self.max_distance
        
        # Updated state representation
        last_signal = [
            self.car.signal1, 
            self.car.signal2, 
            self.car.signal3, 
            orientation,
            normalized_distance  # Added distance to destination
        ]
        
        action = model.update(self.last_reward, last_signal)
        self.scores.append(model.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        self.ball1.pos = self.car.sensor_1
        self.ball2.pos = self.car.sensor_2
        self.ball3.pos = self.car.sensor_3

        # Get car position with bounds checking
        h, w = sand.shape[0], sand.shape[1]
        x = max(0, min(int(self.car.x), w-1))
        y = max(0, min(int(self.car.y), h-1))
        
        # Check a larger area under the car (5x5 instead of single pixel)
        car_area = sand[max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)]
        sand_ratio = np.mean(car_area) if car_area.size > 0 else 1.0
        
        # Improved reward system
        if sand_ratio > 0.4:  # On sand
            self.car.velocity = Vector(3, 0).rotate(self.car.angle)  # Reduced speed on sand
            self.last_reward = -1  # Less harsh penalty
        else:  # On road
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            self.last_reward = 0.2  # Increased reward for being on road
            
            # Small reward for moving toward destination
            direction_reward = 0.05 * (1 - abs(orientation))
            self.last_reward += max(0, direction_reward)

        # Anti-stuck mechanism
        if abs(self.car.velocity_x) < 0.5 and abs(self.car.velocity_y) < 0.5:
            self.stuck_timer += 1
            if self.stuck_timer > 30:  # Shorter stuck detection
                self.last_reward = -1.5
                # Gentle nudge instead of random jump
                self.car.velocity = Vector(6, 0).rotate(self.car.angle + randint(-45, 45))
                self.stuck_timer = 0
        else:
            self.stuck_timer = 0

        # Border handling
        border_buffer = 15
        border_penalty = -0.5
        if self.car.x < border_buffer:
            self.car.x = border_buffer
            self.last_reward = border_penalty
        if self.car.x > self.width - border_buffer:
            self.car.x = self.width - border_buffer
            self.last_reward = border_penalty
        if self.car.y < border_buffer:
            self.car.y = border_buffer
            self.last_reward = border_penalty
        if self.car.y > self.height - border_buffer:
            self.car.y = self.height - border_buffer
            self.last_reward = border_penalty

        # Destination reward
        destination_threshold = 50
        if distance < destination_threshold and not self.destination_rewarded:
            self.last_reward += 5.0  # Reduced destination reward
            self.x_destination = self.width - self.x_destination
            self.y_home = self.height - self.y_home
            self.destination_rewarded = True
        
        if distance > destination_threshold * 1.5:
            self.destination_rewarded = False
        
        # Auto-save
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            print(f"Auto-saving model after {self.save_interval//60} minutes")
            model.save()
            self.last_save_time = current_time
            plt.plot(self.scores)
            plt.title(f'Training Progress - {time.strftime("%Y-%m-%d %H:%M")}')
            plt.savefig(f'training_progress_{int(self.last_save_time)}.png')
            plt.close()

class MyPaintWidget(Widget):
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            if 0 <= last_y < sand.shape[0] and 0 <= last_x < sand.shape[1]:
                sand[last_y, last_x] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1
            density = n_points / length
            touch.ud['line'].width = int(20 * density + 1)
            if 10 <= y < sand.shape[0]-10 and 10 <= x < sand.shape[1]-10:
                # Smaller sand drawing area (5x5)
                sand[y-2:y+3, x-2:x+3] = 1
            last_x = x
            last_y = y

class CarApp(App):
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='Clear', size_hint=(None, None), size=(100, 50), pos=(0, 0))
        savebtn = Button(text='Save', size_hint=(None, None), size=(100, 50), pos=(100, 0))
        loadbtn = Button(text='Load', size_hint=(None, None), size=(100, 50), pos=(200, 0))
        next_track_btn = Button(text='Next Track', size_hint=(None, None), size=(100, 50), pos=(300, 0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        next_track_btn.bind(on_release=lambda btn: parent.switch_track())
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        parent.add_widget(next_track_btn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((int(self.root.height), int(self.root.width)))

    def save(self, obj):
        print('Saving model...')
        model.save()
        plt.plot(self.root.scores)
        plt.title(f'Training Progress - {time.strftime("%Y-%m-%d %H:%M")}')
        plt.savefig(f'training_progress_{int(time.time())}.png')
        plt.show()

    def load(self, obj):
        print('Loading model...')
        model.load()

if __name__ == "__main__":
    CarApp().run()