#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:14:26 2020

@author: jamesark
"""
import numpy
import matplotlib.pyplot as plt


def planet_period(planet):
    """Return orbital period
    Uses planets.csv
    """
    lines = open('planets.csv').readlines()
    for line in lines:
        parts = line.split(',')
        if parts[0] == planet:
            return float(parts[2])


def load_planet_data(file_name):
    """Return matrix with orbital radius and period"""
    rows = []
    lines = open(file_name).readlines()
    for line in lines[1:]:
        parts = line.split(',')
        rows.append((float(parts[1]), float(parts[2])))
    return numpy.array(rows)


def au_to_meters(value):
    return value * 1.496e11


def earth_years_to_seconds(value):
    return value * 3.16e7


def getOrbitalVelocity(r, t):
    return (2 * numpy.pi * r) / t


def getSunMass(v, r):
    G = 6.67e-11
    return ((v ** 2) * r) / G


# Exercise_1.5
# -----------------------------------------------------------------------------------------SUN
numpy.set_printoptions(precision=2)
data = load_planet_data('planets.csv')
data = numpy.array([au_to_meters(data[:, 0]), earth_years_to_seconds(data[:, 1])])
data = numpy.array([data[0, :], getOrbitalVelocity(data[0, :], data[1, :])])
sun_mass = getSunMass(data[1, :], data[0, :])
data = numpy.array([numpy.mean(sun_mass), numpy.std(sun_mass)])
print(data)

# ----------------------------------------------------------------------------------------PLOT
data2 = load_planet_data('planets.csv')
x, y = (data2[:, 0], data2[:, 1])
coefs = numpy.polyfit(x, y, 2)
pxs = numpy.linspace(0, max(x))
poly = numpy.polyval(coefs, pxs)

plt.figure(1, figsize=(12, 8), frameon=False)
plt.plot(x, y, '.r')
plt.plot(pxs, poly, '-')
plt.axis([0, max(x) + 10, -50, max(y) + 10])
plt.title('Degree: 2')
plt.savefig('1.5.png')
plt.close()
