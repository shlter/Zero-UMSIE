import cv2
import torch
import numpy as np

device = 'cuda'


lr = 0.001


illu_factor = 1
reflect_factor = 1
noise_factor = 5000
reffac = 1
gamma = 0.4


g_kernel_size = 5
g_padding = 2
sigma = 3

