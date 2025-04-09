import cv2
import numpy as np
from ex2_utils import *
from ex3_utils import *

class cf_tracker(Tracker):

    def __init__(self, sigma=1.5, scaling_parameter=1, alpha=0.15):
        self.sigma = sigma
        self.alpha = alpha
        self.scaling_parameter = scaling_parameter

        self.target_width = None
        self.target_height = None
        self.region_width = None
        self.region_height = None
        self.x = None
        self.y = None
        self.gauss_peak = None
        self.hanning_window = None
        self.G_hat = None
        self.H_hat_conjugate = None
        

    def name(self):
        return "cf"

    def initialize(self, image, region):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = (image - np.mean(image))/np.std(image)


        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
            
        position = (region[0] + region[2] / 2, region[1] + region[3] / 2) # X x Y
        self.x = position[0]
        self.y = position[1]

        self.target_width = region[2] if region[2] %2 == 1 else region[2] + 1
        self.target_height = region[3] if region[3] %2 == 1 else region[3] + 1

        self.region_width = (math.floor((region[2] * self.scaling_parameter) / 2) * 2) + 1
        self.region_height = (math.floor((region[3] * self.scaling_parameter) / 2) * 2) + 1

        template, mask = get_patch(image, position, (self.region_width, self.region_height))
        
        self.gauss_peak = create_gauss_peak((self.region_width, self.region_height), self.sigma)
        self.hanning_window = create_cosine_window((self.region_width, self.region_height))

        P = (template * self.hanning_window)

        P_hat = np.fft.fft2(P)
        P_hat_conjugate = np.conjugate(P_hat)
        self.G_hat = np.fft.fft2(self.gauss_peak)

        self.H_hat_conjugate = (self.G_hat * P_hat_conjugate) / ((P_hat * P_hat_conjugate)+2000)

    def track(self, image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im = (im - np.mean(im))/np.std(im)

        template, mask = get_patch(im, (self.x, self.y), (self.region_width, self.region_height))

        P = (template * self.hanning_window)

        P_hat = np.fft.fft2(P)
        P_hat_conjugate = np.conjugate(P_hat)
        R = np.fft.ifft2(self.H_hat_conjugate * P_hat).real

        y, x = np.unravel_index(R.argmax(), R.shape)

        if x > self.region_width / 2:
            x = x - self.region_width

        if y > self.region_height / 2:
            y = y - self.region_height

        self.x += x
        self.y += y

        new_H_hat_conjugate = (self.G_hat * P_hat_conjugate) / ((P_hat * P_hat_conjugate)+2000)

        self.H_hat_conjugate = ((1-self.alpha) * self.H_hat_conjugate) + (self.alpha * new_H_hat_conjugate)

        return(round(self.x - (self.target_width/2)), round(self.y - (self.target_height/2)), self.target_width, self.target_height)