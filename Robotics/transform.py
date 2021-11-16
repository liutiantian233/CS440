# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
# from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *
import os


def transformToMaze(alien, goals, walls, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.

        Args:
            alien (Alien): alien instance
            goals (list): [(x, y, r)] of goals
            walls (list): [(startx, starty, endx, endy)] of walls
            window (tuple): (width, height) of the window

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    begin_config = alien.get_config()
    rows = int(window[0] / granularity + 1)
    cols = int(window[1] / granularity + 1)
    maze_map = np.full((rows, cols, 3), SPACE_CHAR)
    offset = [0, 0, 0]

    for index in np.ndindex(rows, cols, 3):
        config = idxToConfig(index, offset, granularity, alien)
        alien.set_alien_config(config)
        not_within = not is_alien_within_window(alien, window, granularity)
        touch_wall = does_alien_touch_wall(alien, walls, granularity)
        touch_goal = does_alien_touch_goal(alien, goals)
        if not_within or touch_wall:
            maze_map[index] = WALL_CHAR
        elif touch_goal:
            maze_map[index] = OBJECTIVE_CHAR

    idx = configToIdx(begin_config, offset, granularity, alien)
    maze_map[idx[0]][idx[1]][idx[2]] = START_CHAR
    return Maze(maze_map, alien, granularity)
