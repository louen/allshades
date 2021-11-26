
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

    # standard square mesh
    tris = []
    (nx,ny) = (k,len(curve))
    for i in range(nx-1):
        for j in range(ny-1):

            index = lambda i,j: i + nx* j;

            tris.append([index(i,j),index(i,j+1), index(i+1, j+1)])
            tris.append([index(i,j),index(i+1,j+1), index(i+1, j)])
    
    return points, tris, nx, ny

    # Triangle layout
    # . ny (points along the Hilbert curve)
    # . 
    # .  (i,j+1) - (i+1,j+1)
    # .  | .  /     |
    # .  (i,j) -- (i+1,j)
    #  -> Z axis (extrusion axis) + order of points ... nx 

# write to files. expect flat lists
def write_obj(file,points,tris):
    with open(file, "w") as f:
        for v in points:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        for t in tris:
            #obj face indices start at 1
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

def write_ply(file, points, tris, colors):
    with open(file, "w") as f:
        f.write("ply\n");
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green \n")
        f.write("property uchar blue \n")
        f.write(f"element face {len(tris)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, c in zip(points, colors):
            f.write(f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
        for t in tris:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")

def animate_flattening(points, width, height, scale, time):
    result = points.reshape(height, width, 3)
    for x in range(height):
        for y in range(width):
            p = result[x,y] 
            v = y / (width -1) # inversion ?
            u = x / (height - 1)

            target = np.array([scale[0] * u, 0.0, scale[1] * v ])

            #linerp for now
            result[x,y] = time * target + (1.0 - time) * p


    return result.reshape(points.shape)

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

#plt.gca().set_aspect(1.0)
#plt.gca().set_xlim([0,1])
#plt.gca().set_ylim([0,1])
#draw_hilbert_curve(plt.gca(),hilbert_curve_points(1), 'r') 
#draw_hilbert_curve(plt.gca(),hilbert_curve_points(2), 'b') 
#draw_hilbert_curve(plt.gca(),hilbert_curve_points(3), 'g') 
p4 = hilbert_curve_points(2)
#draw_hilbert_curve(plt.gca(),p4, 'c') 

pts = np.array([hilbert_curve_param(p4, t) for t in np.linspace(0,1,127)])
x = pts[:,0]
y = pts[:,1]
#plt.gca().scatter(x,y)

v,t, width, height = hilbert_carpet(2,4)
#write_obj("test.obj", v,t)

c =  (255.0 * v).astype('uint8')
write_ply("test.ply", v, t, c)


for time in np.linspace(0,1,11):
    vt = animate_flattening(v,width, height, (5.0, 1.0), time)
    write_ply(f"test{time:.3f}.ply", vt, t,c)
#plt.show()

 

