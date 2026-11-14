"""
Measuring how well the camera can see, straight from the image.

These numbers say nothing about motion. They describe the picture itself.
The idea behind Step 4 is that when the camera is struggling (dark, rain on
the lens, blur) these numbers move, and when the CAN bus is being attacked
they do not, because an attacker on the CAN bus cannot touch the image.

They are computed in the same pass as the motion estimate and saved with
it, so the images do not have to be kept around afterwards.

Raw values differ a lot between scenes for reasons that have nothing to do
with degradation: a street full of signs and windows is "sharper" than an
empty road. Later steps compare each value against the same scene's opening
seconds instead of using it directly (brief section 6, lesson 3).
"""

import cv2
import numpy as np


def frame_appearance(gray):
    """
    Three simple measurements of one grayscale frame.

    brightness : average pixel value, 0 (black) to 255 (white).
    contrast   : how spread out the pixel values are. Low at night and in
                 fog, when everything is a similar shade.
    sharpness  : how much fine detail there is. The Laplacian responds to
                 edges, so its variance is high for a crisp image and low
                 for a blurred or featureless one.
    """
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return {
        "brightness": float(small.mean()),
        "contrast": float(small.std()),
        "sharpness": float(cv2.Laplacian(small, cv2.CV_64F).var()),
    }
