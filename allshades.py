
#!/usr/bin/python3

import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image

# recursive implementation of hilbert curve points
# points are normalized in [0,1]x[0,1]
def hilbert_curve_points(n):
    if n == 1:
        return np.array([
            [0.25, 0.25],
            [0.25,0.75],
            [0.75, 0.75],
            [0.75, 0.25]
        ])
    else:
        # The curve of order n is made of 4 copies
        # of the curve of order n-1
        curve_n1 = hilbert_curve_points(n-1)
            
        r_left = np.array([[0,1],[-1,0]]) # 90 degrees left rotation
        r_right = np.array([[0,-1], [1,0]]) # 90 degrees right

        return np.concatenate([
        np.flip(np.matmul(r_left, (0.5 * (curve_n1 - [0.5, 0.5])).T).T + [0.25, 0.25], axis = 0),
        0.5 * (curve_n1) + [0.0,0.5],
        0.5 * (curve_n1) + [0.5,0.5],
        np.flip(np.matmul(r_right, (0.5 * (curve_n1 - [0.5, 0.5])).T).T + [0.75, 0.25], axis = 0)
        ])


def hilbert_carpet(n,k):
    curve = hilbert_curve_points(n)
    
    points = np.array([
        [x[0][0],x[0][1],x[1]] for x in itertools.product(curve, np.linspace(0,1,k))
    ])

    
    


    

def hilbert_curve_param(points, t):
    num_segments = points.shape[0] - 1
    if t <= 0:
        return points[0]
    elif t >= 1.0:
        return points[-1]
    
    segment_idx = int(np.floor(t * num_segments))
    assert(segment_idx < num_segments)

    t0 = segment_idx / num_segments
    r = num_segments * (t - t0)

    a = points[segment_idx]
    b = points[segment_idx+1]

    return a + r * (b-a)


def draw_hilbert_curve(ax, points, color='b'):
    x = points[:,0]
    y = points[:,1]
    ax.plot(x,y,marker='.', c=color)

def red(_):
    return (128, 0, 0)

def hue1d(index, axis=0):
    t = index[axis]
    return colors.hsv_to_rgb([t,1,1])

def float_array_to_img(array):
    assert(len(array.shape) == 3)
    return Image.fromarray((255.0 *array).astype('uint8')) 


def flag(h, w, sampler):
    img_array = np.zeros((h,w,3))
    for ((i,j), _) in np.ndenumerate(img_array[:,:,0]):
        xy = (i / h, j / w)
        img_array[i, j, :] = sampler(xy)
    float_array_to_img(img_array).show()


#flag(100,200,red)
#flag(100,200,hue1d)

plt.gca().set_aspect(1.0)
plt.gca().set_xlim([0,1])
plt.gca().set_ylim([0,1])
#draw_hilbert_curve(plt.gca(),hilbert_curve_points(1), 'r') 
#draw_hilbert_curve(plt.gca(),hilbert_curve_points(2), 'b') 
#draw_hilbert_curve(plt.gca(),hilbert_curve_points(3), 'g') 
p4 = hilbert_curve_points(4)
draw_hilbert_curve(plt.gca(),p4, 'c') 

pts = np.array([hilbert_curve_param(p4, t) for t in np.linspace(0,1,127)])
x = pts[:,0]
y = pts[:,1]
plt.gca().scatter(x,y)

hilbert_carpet(1,3)

plt.show()

 

