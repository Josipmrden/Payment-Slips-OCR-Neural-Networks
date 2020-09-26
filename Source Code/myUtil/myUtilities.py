import cv2
import math
import numpy as np

def see_if_rotated(box, base_x):
    rotated = True
    j = 0
    for b in box:
        if b[0] == base_x:
              j = j + 1
        if j == 2:
            rotated = False
            break
    return rotated

def find_base(box):
    base_x = -1
    base_index = -1
    i = 0
    for b in box:
        if i == 0:
            base_x = b[0]
            base_index = i
        elif b[0] < base_x:
            base_x = b[0]
            base_index = i
        i = i + 1
    base = (box[base_index][0], box[base_index][1])

    return base, base_x, base_index

def find_biggest_rect(contours):
    area = 0
    x1 = 0
    y1 = 0
    w1 = 0
    h1 = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        tempArea = w * h
        if tempArea > area:
            x1, y1, w1, h1 = x, y, w, h
            area = tempArea
            cout = c

    return cout
    # cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

def rectangle_details(box, rotated):
    base, _, base_index = find_base(box)
    if rotated == False:
        return not_rotated_details(base, base_index, box)
    else:
        return rotated_details(base, base_index, box)

def not_rotated_details(base, base_index, box):
    return -1, -1, 0

def rotated_details(base, base_index, box):
    w, h, angle = -1, -1, -1
    list = getListFromBox(box)
    up, up_list = get_elements_up(base, list)
    if up == 1:
        highest = up_list[0]
        w = distance(base[0], base[1], highest[0], highest[1])
        l3 = find_lowest(list)
        h = distance(base[0], base[1], l3[0], l3[1])
        angle = calc_angle(base[0], base[1], highest[0], highest[1])
    else:
        highest = find_highest(list)
        h = distance(base[0], base[1], highest[0], highest[1])
        lowest = find_lowest(list)
        w = distance(base[0], base[1], lowest[0], lowest[1])
        angle = calc_angle(base[0], base[1], lowest[0], lowest[1])

    return w, h, angle

def getListFromBox(box):
    list = []
    for b in box:
        list.append((b[0], b[1]))
    return list

def get_elements_up(base, list):
    up = 0
    up_list = []
    for l in list:
        if l[0] == base[0] and l[1] == base[1]:
            continue
        elif l[1] < base[1]:
            up = up + 1
            up_list.append(l)

    return up, up_list

def distance(x1, y1, x2, y2):
    x = (x2 - x1)*(x2 - x1)
    y = (y2 - y1)*(y2 - y1)

    return math.sqrt(x + y)

def find_highest(list):
    high = (-1, -1)
    for l in list:
        if high == (-1, -1):
            high = l
            continue
        if l[1] < high[1]:
            high = l

    return high

def find_lowest(list):
    low = (-1, -1)
    for l in list:
        if low == (-1, -1):
            low = l
            continue
        if l[1] > low[1]:
            low = l

    return low

def calc_angle(x1, y1, x2, y2):
    v1 = (x1, y1)
    v2 = (x2, y2)

    v1_theta = math.atan2(v1[1], v1[0])
    v2_theta = math.atan2(v2[1], v2[0])

    r = v2_theta - v1_theta

    if r < 0:
        r += 2+math.pi

    return r