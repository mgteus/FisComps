# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:05:28 2021

@author: mgteus

"""


import glob
from PIL import Image

# filepaths
fp_in = r"Potts\img\W-q9_0721\fig*.png"
fp_out = "Potts\gifs\W-q9_0721.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=500, loop=0)