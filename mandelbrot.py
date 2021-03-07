import numpy as np
import cv2

STEP = 1.24


def get_iter(c: complex, thresh: int, max_steps: int) -> int:
    z = c
    i = 0
    while i < max_steps and (z*z.conjugate()).real < thresh:
        z = z*z + c
        i += 1
    return i


def plotter(center: list, resolution: tuple, zoom: float, thresh: float, max_steps: int) -> np.array:
    ratio = (1, resolution[0]/resolution[1]) if resolution[1] > resolution[0] else (resolution[1]/resolution[0], 1)
    real_range = np.array([center[0] - (STEP/zoom), center[0] + (STEP/zoom)])*ratio[0]
    imag_range = np.array([center[1] - (STEP/zoom), center[1] + (STEP/zoom)])*ratio[1]
    real_axis = np.linspace(real_range[0], real_range[1], num=resolution[1], dtype=np.float32)
    imag_axis = np.linspace(imag_range[0], imag_range[1], num=resolution[0], dtype=np.float32)

    complex_grid = np.zeros((resolution[0], resolution[1]), dtype=np.complex64)
    complex_grid.real, complex_grid.imag = np.meshgrid(real_axis, imag_axis)
    z_grid = np.zeros_like(complex_grid)

    img = np.zeros((resolution[0], resolution[1]), dtype=np.uint8)
    todo_grid = np.ones((resolution[0], resolution[1]), dtype=bool)

    for i in range(max_steps):
        z_grid[todo_grid] = z_grid[todo_grid]**2 + complex_grid[todo_grid]
        mask = np.logical_and((z_grid.real**2 + z_grid.imag**2) > thresh, todo_grid)
        img[mask] = int(i * (255/max_steps))
        todo_grid = np.logical_and(todo_grid, np.logical_not(mask))
    return img


resolution = (800, 1200)
center = [-0.76, 0]
zoom = 1

while 1:
    img = plotter(center=center, zoom=zoom, resolution=resolution, thresh=3, max_steps=256)
    cv2.imshow('', img)
    cv2.waitKey(1)

    command = input('[W] - up\n[S] - down\n[A] - left\n[D] - right\n[Z] - zoom in\n[X] - zoom out')
    if command == 'w':
        center[1] -= 0.15 * (STEP/zoom)  # move up 10% of the current view
    elif command == 's':
        center[1] += 0.15 * (STEP/zoom)  # move down 10% of the current view
    elif command == 'a':
        center[0] -= 0.15 * (STEP/zoom)  # move left 10% of the current view
    elif command == 'd':
        center[0] += 0.15 * (STEP/zoom)  # move right 10% of the current view
    elif command == 'z':
        zoom /= 0.8
    elif command == 'x':
        zoom *= 0.8
    else:
        print('Wrong command. Try again.')
