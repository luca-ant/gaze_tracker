# Gaze tracker


<br />
<br />
### Calibration phase
<p align="center">
  <img width=1024px src="https://github.com/luca-ant/gaze_tracker/blob/master/videos/gaze_tracker_calibration.gif">
</p>

### Mouse control phase
<p align="center">
  <img width=1024px src="https://github.com/luca-ant/gaze_tracker/blob/master/videos/gaze_tracker_mouse.gif">
</p>

## Getting started

* Clone repository
```
git clone https://github.com/luca-ant/gaze_tracker.git
```

* Install dependencies
```
sudo apt install python3-setuptools python3-pip python3-venv
```
or
```
sudo pacman -S python-setuptools python-pip python-virtualenv
```

* Create a virtual environment and install requirements modules
```
cd gaze_tracker
python3 -m venv venv
source venv/bin/activate

python3 -m pip install -r requirements.txt
```

## Running

Simply move to `gt` folder and run `main.py`

```
cd gt
python main.py
```
