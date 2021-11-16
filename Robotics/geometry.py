# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP2
"""

import math
import numpy as np
from alien import Alien


def line_length(x1, y1, x2, y2):
    return math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))


def point_to_line_distance(point, line, radius, granularity=0):
    bound = granularity / math.sqrt(2)
    px, py = point
    x1, y1, x2, y2 = line
    line_distance = line_length(x1, y1, x2, y2)
    if line_distance == 0:
        return False
    u = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1))) / math.pow(line_distance, 2)
    if u <= 0:
        return line_length(px, py, x1, y1) - radius - bound <= 0
    elif u >= 1:
        return line_length(px, py, x2, y2) - radius - bound <= 0
    else:
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        return line_length(px, py, ix, iy) - radius - bound <= 0


def does_alien_touch_wall(alien, walls, granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
            [(startx, starty, endx, endx), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """
    if alien.is_circle():
        center = alien.get_centroid()
        for wall in walls:
            if point_to_line_distance(center, wall, alien.get_width(), granularity):
                return True
    else:
        head_and_tail = alien.get_head_and_tail()
        for wall in walls:
            if np.isclose(head_and_tail[0][0], head_and_tail[1][0]):
                point_y = min(head_and_tail[0][1], head_and_tail[1][1])
                while point_y <= max(head_and_tail[0][1], head_and_tail[1][1]):
                    if point_to_line_distance((head_and_tail[0][0], point_y), wall, alien.get_width(), granularity):
                        return True
                    point_y += 1
            else:
                point_x = min(head_and_tail[0][0], head_and_tail[1][0])
                while point_x <= max(head_and_tail[0][0], head_and_tail[1][0]):
                    if point_to_line_distance((point_x, head_and_tail[0][1]), wall, alien.get_width(), granularity):
                        return True
                    point_x += 1
    return False


def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...].
            There can be multiple goals

        Return:
            True if a goal is touched, False if not.
    """
    if alien.is_circle():
        center = alien.get_centroid()
        for goal in goals:
            if math.sqrt(math.pow((goal[0] - center[0]), 2) + math.pow((goal[1] - center[1]), 2)) <=\
                    goal[2] + alien.get_width():
                return True
    else:
        for goal in goals:
            line = (alien.get_head_and_tail()[0][0], alien.get_head_and_tail()[0][1],
                    alien.get_head_and_tail()[1][0], alien.get_head_and_tail()[1][1])
            if point_to_line_distance((goal[0], goal[1]), line, goal[2] + alien.get_width()):
                return True
    return False


def is_alien_within_window(alien, window, granularity):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """
    walls = [(0, 0, window[0], 0),
             (0, 0, 0, window[1]),
             (0, window[1], window[0], window[1]),
             (window[0], 0, window[0], window[1])]
    return not does_alien_touch_wall(alien, walls, granularity)
