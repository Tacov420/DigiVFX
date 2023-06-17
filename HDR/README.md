# README

Dependencies

- tqdm
- cv2
- random
- matplotlib
- numpy

Usage(Run on default set of photos)

1. Run `python3 main.py`

Usage(Run on different set of photos)

1. Around line 218, enter all the shutter speed.
2. Around line 205, enter all name of images
3. Run `python3 main.py`

Result 

- The result image from Reinhard tone mapping would be `result_reinhard.jpg` at the same location of the terminal.
- The result image from Adaptive Logarithmic Mapping would be `result_log_{base}.jpg` at the same location of the terminal.