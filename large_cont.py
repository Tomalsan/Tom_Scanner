import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
# from biga import start_animation
from canny import perform_canny

from math import inf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

SPREAD_VAL = 40  

def spread(image, sp_x, sp_y, to_replace, replace_with, if_not = False):
    h, w = image.shape
    
    parent_map = {}
    length = 0
    last = None

    stack = [(sp_x, sp_y, 0)]
    parent_map[(sp_x, sp_y)] = None
    
    while stack:
        x, y, it = stack.pop()
        
        if( if_not ):
            if image[x, y] == to_replace:
                continue
        else:
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
            if 0 <= nx < h and 0 <= ny < w and ( (not if_not and image[nx, ny] == to_replace) or (if_not and image[nx, ny] != to_replace)) :
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


def spread_make_zero(image, sp_x, sp_y):
    h, w = image.shape
    
    parent_map = {}    
    stack = [(sp_x, sp_y, 0)]
    parent_map[(sp_x, sp_y)] = None
    
    while stack:
        x, y, it = stack.pop()
        
        indices = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        for dx, dy in indices:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and image[nx, ny] != 0:
                if (nx, ny) not in parent_map: # Check if not already visited
                    image[nx,ny] = 0
                    parent_map[(nx, ny)] = (x, y)
                    stack.append((nx, ny, it))
    return image

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

                temp = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
                indices = []
                for pt in temp:
                    indices.append(pt)
                    # indices.append((pt[0]*2, pt[1]*2))
                
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

    largest_contour = []
    for cnt in contours:
        if(len(cnt) < 200):
            continue
        
        # segments = find_line_segments(cnt)
        # for seg in segments:
        #     if len(seg) < 200:
        #         continue
        for x, y in cnt:
            largest_contour.append((x,y))


    # largest_contour = max(contours, key=len) if contours else []

    return largest_contour

def get_edge_points(image):
    largest_contour = get_contours(image)
    print(f"Largest contour size: {len(largest_contour)}")
        
    W_F = H_F = 1
    points = []
    for pt in largest_contour:
        points.append((pt[0], pt[1]))
    return points


def show_image(name, image, wait=True):
    cv2.imshow(name, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def extract_page(image):
    
    copied = image.copy()
    h,w = image.shape
    
    areas = []
    
    for x in range(h):
        for y in range(w):
            if(image[x,y] == 0):
                continue
            
            points = spread(image, x, y, to_replace = 0, replace_with = 0, if_not=True)
            
            areas.append(points)
    
    areas =  sorted(areas, key=len, reverse=True)
    
    for i in range(1, len(areas)):
        area = areas[i]
        for pt in area:
            copied = spread_make_zero(copied, pt[0], pt[1])
    return copied

def crop_black(image, image_color):
    
    min_x, min_y = 1000, 1000
    max_x, max_y = -1000, -1000
    
    h,w = image.shape
    for y in range(h):
        for x in range(w):
            if(image[y,x] == 0):
                continue
            
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    
    nh = max_x - min_x + 1
    nw = max_y - min_y + 1
    
    # res = np.zeros((nh,nw,3), dtype=np.uint8)
    # for y in range(min_x, max_x):
    #     for x in range(min_y, max_y):
    #         res[x-min_x, y-min_y] = image_color[x,y]
    
    res = image_color[min_y:max_y+1, min_x:max_x+1]

    return res, (min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y) # tl, tr, br, bl


def sortCorners(corners):
    corners = corners.reshape((4,2))
    # create empty new corners
    sortedCorners = np.zeros((4,1,2),np.int32)
    # sum each corner width + height => smallest is at the top left[0,0]
    # biggest will be bottom right [width, height]
    # we want to make the corner like this order [[0, 0], [width, 0], [0, height], [width, height]]
    add = corners.sum(1)
    # smallest
    sortedCorners[0] = corners[np.argmin(add)]
    # biggest
    sortedCorners[3] = corners[np.argmax(add)]
    # The difference between width - height for the remaining points:
    #  if the result (height - width) is negative(minimum) => will be at [width, 0]
    #  if the result (height - width) is positive(maximum) => will be at [0, height]
    diff = np.diff(corners,axis=1)
    sortedCorners[1]= corners[np.argmin(diff)]
    sortedCorners[2] = corners[np.argmax(diff)]
    
    return sortedCorners

def apply_warp(image, tl, tr, br, bl):
    
    height, width, c = image.shape
    
    corners = np.array([tl,tr,br,bl])
    corners = sortCorners(corners)
    
    # set the points
    points1 = np.float32(corners)
    points2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Wrapper transform computer transform matrix
    matrix = cv2.getPerspectiveTransform(points1, points2)
    # result
    result = cv2.warpPerspective(image, matrix, (width, height))
    
    return result

def save_edge_image(edge_image, folder='Edge'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    existing_files = [int(f.split('.')[0]) for f in os.listdir(folder) if f.endswith('.png') and f.split('.')[0].isdigit()]
    next_index = max(existing_files) + 1 if existing_files else 1
    
    output_file = os.path.join(folder, f"{next_index}.png")
    cv2.imwrite(output_file, edge_image)
    return output_file


def save_output_image(edge_image, folder='Output'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    existing_files = [int(f.split('.')[0]) for f in os.listdir(folder) if f.endswith('.png') and f.split('.')[0].isdigit()]
    next_index = max(existing_files) + 1 if existing_files else 1
    
    output_file = os.path.join(folder, f"{next_index}.png")
    cv2.imwrite(output_file, edge_image)
    return output_file

def draw_bounding_box(image_path):
    
    image = cv2.imread(image_path, 0)
    image_color = cv2.imread(image_path)
    height, width = 512, 512
    resized_image = cv2.resize(image, (height, width))
    image_color = cv2.resize(image_color, (height, width))
    inp = resized_image.copy()
    edge = perform_canny(image=resized_image, show=False)
    
    # show_image("Edge Detected Image", image=edge)
    
    output_file = save_edge_image(edge)
    print(f"Image saved as {output_file}")
    
    
    # edge = cv2.imread('output7.png',0)
    # cv2.resize(edge, (512,512))

    edge_points = get_edge_points(edge)
    
    
    #Segementation
    for x in range(height):
        for y in range(width):
            if (x, y) in edge_points:
                break
            nx = x+10 
            ny = y+10 
            if 0 <= nx < height and 0 <= ny < width:
                resized_image[nx,ny] = 0
        for y in range(width-1,0,-1):
            if (x, y) in edge_points:
                break
            nx = x+10 
            ny = y+10
            if 0 <= nx < height and 0 <= ny < width:
                resized_image[nx,ny] = 0
    
    page = extract_page(resized_image.copy())
    
    only_page, tl, tr, br, bl = crop_black(page, image_color)
    
    warped = apply_warp(image_color.copy(), tl, tr, br, bl)
    
    output_file = save_output_image(warped)
    print(f"Image saved as {output_file}")
    

    cv2.imshow('Page',page)
    cv2.waitKey(0)
    cv2.imshow('Page Cropped',only_page)
    cv2.waitKey(0)
    cv2.imshow('Warped',warped)
    cv2.waitKey(0)
    cv2.imshow('Edge',edge)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    