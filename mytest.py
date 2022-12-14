import imageio
import os
import numpy as np
image_list = []
def compose_gif(directory_name):
    images_name = np.sort(os.listdir(directory_name))
    for filename in images_name:
        print(filename)
        img = imageio.imread(directory_name + "/" + filename)
        image_list.append(img)
    imageio.mimsave("depth.gif",image_list, fps=30)
    
if __name__ == "__main__":
    compose_gif('/share/NeRF/nerf_pl/results/llff/depth')