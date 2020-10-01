#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:14:26 2020

@author: jamesark
"""
def planet_period(planet):
    """Return orbital period
    Uses planets.csv
    """
    lines = open('planets.csv').readlines()
    for line in lines:
        parts = line.split(',')
        if parts[0] == planet:
            return float(parts[2])



import numpy
import matplotlib.pyplot as plt

def load_planet_data(file_name):
    """Return matrix with orbital radius and period"""
    rows = []
    lines = open(file_name).readlines()
    for line in lines[1:]:
        parts = line.split(',')
        rows.append( (float(parts[1]), float(parts[2])) )
    return numpy.array(rows)

data = load_planet_data('planets.csv')

plt.plot(data[:,0],data[:,1],'x')

