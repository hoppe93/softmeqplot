#!/usr/bin/env python3
#
# A simple example of how to use
# the SyntheticImage class. In
# difference to the main softviz
# program, this example does not
# require Qt.
##################################

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from MeqPlot import MeqPlot

# Set figure size
plt.figure(figsize=(5,7))

# Create GRI image object
mp = MeqPlot()

# Load the SOFT image
mp.loadDataFile('../../../runaway/mathias/SOFT/resources/C-Mod/C-Mod.mat')

# Plot the image
mp.plotBr, mp.plotBz = True, True
mp.calculateFluxSurfaces()

mp.assemblePlot()
mp.plotWallCrossSection(linewidth=3)
mp.plotFluxSurfaces(plotstyle='r:')
mp.plotSeparatrix(plotstyle='r--')

plt.show()

