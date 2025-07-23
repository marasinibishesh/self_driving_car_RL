# Self-Driving Car Simulation with Reinforcement Learning

This project shows how a car can learn to drive itself using artificial intelligence! The car starts with no knowledge and learns through trial-and-error to navigate tracks, avoid obstacles, and reach destinations.

## What is Reinforcement Learning?

Reinforcement Learning (RL) is how AI learns from experience:
1. The car **observes** its environment (using sensors)
2. It **takes actions** (steering left/right or going straight)
3. It **receives rewards** (+ for good actions, - for mistakes)
4. It **improves** its strategy over time

It's like teaching a child to ride a bike - they learn from successes and falls!

## What You'll See in This Simulation

- 🚗 A car that learns to drive on different tracks
- 🛣️ 4 challenging race tracks
- 🧠 An AI "brain" that improves as it practices
- 📊 Performance graphs showing learning progress


## System Requirements

- Windows/Mac/Linux computer
- 4GB RAM minimum (8GB recommended)
- 500MB free disk space
- No coding experience needed!

## Step-by-Step Installation Guide

### 1. Install Python
- Go to [python.org](https://python.org)
- Click "Downloads"
- Install Python 3.10 (check "Add Python to PATH" during installation)

### 2. Download Project Files
[Download ZIP File](https://example.com/car-simulation.zip) and extract to:
- `C:\SelfDrivingCar` (Windows) or
- `/Users/YourName/SelfDrivingCar` (Mac)

### 3. Install Required Software
Open **Command Prompt** (Windows) or **Terminal** (Mac/Linux):

```bash
pip install numpy matplotlib pillow torch kivy

```
## 4. Add Track Images

Create a folder named `tracks` inside your project directory and add these 8 images:

* `track1.png`, `track1-overlay.png`
* `track2.png`, `track2-overlay.png`
* `track3.png`, `track3-overlay.png`
* `track4.png`, `track4-overlay.png`

---

## 🚀 How to Run the Simulation

1. Open **Command Prompt** / **Terminal**
2. Navigate to your project directory:
   `cd C:\SelfDrivingCar` *(or your folder path)*
3. Run the simulation:
   `python main.py`

---

## 🎮 Simulation Controls

| Button         | Function                            |
| -------------- | ----------------------------------- |
| **Clear**      | Reset all drawn obstacles           |
| **Save**       | Save the AI's learned knowledge     |
| **Load**       | Load previous driving experience    |
| **Next Track** | Switch to next racing track         |
| **Mouse**      | Click/drag to create sand obstacles |

---

## 🤖 What Happens During Training

**First 5 Minutes:**

* Car drives randomly
* Makes many mistakes
* Learns basic road recognition

**5–20 Minutes:**

* Starts avoiding sand traps
* Develops steering control
* Reaches destinations sometimes

**20+ Minutes:**

* Navigates tracks efficiently
* Avoids all obstacles consistently
* Adapts to different tracks

---

## ⚙️ Customization Options

**Create New Tracks:**

* Make `800x600` pixel images
* **Black = Road**, **White = Obstacles**
* Save as `track5.png` and `track5-overlay.png`

**Adjust Difficulty:**
Edit `main.py` in Notepad or any code editor:

* Line 157: `last_reward = -2` *(penalty for sand)*
* Line 212: `last_reward = 10` *(reward for reaching destination)*

**Change Learning Speed:**
In `main.py`, modify the following line:
`gamma_i = 0.9` *(Use 0.8 for fast learner, 0.95 for careful thinker)*

---

## 🛠️ Troubleshooting Guide

| Problem                 | Solution                                     |
| ----------------------- | -------------------------------------------- |
| Missing track images    | Create `tracks` folder with 8 PNG files      |
| Python not found        | Reinstall Python with "Add to PATH" option   |
| Library errors          | Run `pip install --upgrade numpy torch kivy` |
| Black screen on startup | Ensure all track images are 800x600 pixels   |
| Car gets stuck          | Click "Clear" to remove obstacles            |

---

## 📚 Learning Resources
* [Official Kivy Guide](https://kivy.org/doc/stable/)
* [PyTorch Neural Networks](https://pytorch.org/tutorials/)

> **🧠 The car will show noticeable improvement after 30 minutes of training — be patient during the early learning phase!**

---

**Developed with ❤️ using Python, PyTorch and Kivy**
**License: [MIT](https://opensource.org/licenses/MIT)**

---
