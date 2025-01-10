import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from canny import perform_canny

SPREAD_VAL = 40  

def spread(image, sp_x, sp_y, to_replace, replace_with):
    h, w = image.shape
    
    parent_map = {}
    length = 0
    last = None

    stack = [(sp_x, sp_y, 0)]
    parent_map[(sp_x, sp_y)] = None
    
    while stack:
        x, y, it = stack.pop()
        if image[x, y] != to_replace:
            continue

        image[x, y] = replace_with
        
        it += 1
        if it > length:
            length = it
            last = (x, y)

        indices = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        for dx, dy in indices:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and image[nx, ny] == to_replace:
                if (nx, ny) not in parent_map: # Check if not already visited
                    parent_map[(nx, ny)] = (x, y)
                    stack.append((nx, ny, it))

    # Get the longest path
    points = []
    while last is not None:
        points.append(last)
        last = parent_map[last]
    points.reverse()

    return points

def get_contours(image):
    image = image.copy()
    h,w = image.shape

    pad = 10
    image = image[pad:h-pad, pad:w-pad]
    h,w = image.shape

    contours = []
    
    it = 1
    visited = {}

    def find_nearest_white_pixel(sx,sy):
        nonlocal visited, it
        to_it = (sx, sy)

        while to_it != None:
            queue = deque()
            queue.append(to_it)

            to_it = None
            count = 0
            while queue:
                x, y = queue.popleft()
                if visited.get((x,y)) == True:
                    continue
                count += 1
                
                image[x, y] = SPREAD_VAL

                indices = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
                for dx, dy in indices:
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= h or ny < 0 or ny >= w or image[nx, ny] == SPREAD_VAL: 
                        continue

                    if image[nx, ny] == 255:
                        to_it = (nx, ny)
                        queue.clear()
                        break
                    if visited.get(to_it) == None:
                        queue.append( (nx,ny) )
                visited[(x,y)] = True

            if to_it == None:
                break
            
            # print(f"Spreading no: {it}")
            it += 1

            points = spread(image, to_it[0], to_it[1], to_replace = 255, replace_with = 2*SPREAD_VAL)
            last_pt = points[ len(points)-1 ]
            
            points = spread(image, last_pt[0], last_pt[1], to_replace = 2*SPREAD_VAL, replace_with = SPREAD_VAL)
            if( len(points) > 25):
                contours.append( points )

            # show_image("Spread result", image=image)
            to_it = points[ len(points) - 1 ]
    
    for x in range(h):
        for y in range(w):
            if visited.get((x,y)) == None:
                find_nearest_white_pixel(x,y)

   
    return contours

def get_edge_points(image):
    contours = get_contours(image)

    print(f"Total contours found: {len(contours)}")
    W_F = H_F = 2

    points = []
    for i in range( len(contours) ):
        cnt = contours[i]
        for pt in cnt:
            points.append( (pt[0]/W_F, pt[1]/H_F) )
    return points

def show_image(name, image, wait=True):
    cv2.imshow(name, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def start(image_path):
    image = cv2.imread(image_path,0)
    height,width = 1024,1024
    resized_image=cv2.resize(image,(height,width))
    resized='resized2.png'
    cv2.imwrite(resized,resized_image)
    edge = perform_canny(image=resized_image, show=False)
    show_image("Edge Detected Image", image=edge)
    output_file = "output2.png"
    cv2.imwrite(output_file,edge)
    edge_points = get_edge_points(edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

