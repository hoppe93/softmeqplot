# SOFT MAGNETIC EQUILIBRIUM PLOT CLASS
#
# This class is a simple interface for plotting SOFT Magnetic Equilibria
# with overlays etc. using Python's matplotlib.
#

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.ticker
import io, os
import subprocess

class MeqPlot:

    def __init__(self, figure=None, canvas=None, registerGeriMap=True):
        # PROPERTIES
        self.canvas = canvas
        self.colormapName = 'GeriMap'
        self.figure = figure
        self.flux = None
        self.overlayWallCrossSection = False
        self.overlaySeparatrix = False
        self.overlayFluxSurfaces = False
        self.overlayMagneticAxis = False

        self.B = None
        self.R, self.Z = None, None
        self.wall = None
        self.separatrix = None
        self.maxis = None
        self.name = ""
        self.description = ""

        self.plotBr = False
        self.plotBphi = False
        self.plotBz = False

        try:
            self.SOFTPATH = os.environ['SOFTPATH']

            # Make sure path does not end with a slash
            if self.SOFTPATH[-1] == '/':
                self.SOFTPATH = self.SOFTPATH[:-1]
        except KeyError:
            print('WARNING: Unable to determine SOFT path')
            self.SOFTPATH = None

        # Internal properties
        self._fluxOverlayHandles = []
        self._magneticAxisHandle = None
        self._orbitHandles = []
        self._separatrixOverlayHandle = None
        self._wallCrossSectionOverlayHandle = None

        if self.figure is None:
            self.figure = plt.gca().figure
            if self.canvas is not None:
                raise ValueError("Canvas set, but no figure given. If no figure is given, no canvas may be given.")
        if self.canvas is None:
            self.canvas = self.figure.canvas

        self.axes = None

        self._meqfile = None
        self._meshR = None
        self._meshZ = None
        self._inputIsHDF5 = False

        if registerGeriMap:
            MeqPlot.registerGeriMap()

    ####################################################
    #
    # GETTERS
    #
    ####################################################
    def hasSeparatrix(self): return self.separatrix is not None
    def getPlasmaBoundaries(self):
        r0 = self.maxis[0]
        if self.separatrix is not None:
            rmax = np.amax(self.separatrix[0,:])
        else:
            rmax = np.amax(self.wall[0,:])
        
        return r0, rmax

    ####################################################
    #
    # SETTERS
    #
    ####################################################
    def setFluxSurfaces(self, flux): self.flux = flux
    def setSeparatrix(self, separatrix): self.separatrix = separatrix
    def setRZ(self, R, Z):
        self.R = R
        self.Z = Z
        self._meshZ, self._meshR = np.meshgrid(Z, R)

    ####################################################
    #
    # PUBLIC METHODS
    #
    ####################################################
    def assemblePlot(self):
        """
        Plot the magnetic equilibrium, applying all settings
        given to this MeqPlot object. This means any overlays
        will be plotted.
        """
        self.clearPlot()
        self.axes = self.figure.add_subplot(111)

        # Reset handles
        self._fluxOverlayHandles = []
        self._magneticAxisHandle = None
        self._orbitHandles = []
        self._separatrixOverlayHandle = None
        self._wallCrossSectionOverlayHandle = None

        # Plot image
        self.plotEq()

        # Plot overlays
        self.plotOverlays()

    def calculateFluxSurfaces(self, normalizedRadii=None, radii=None):
        """
        Runs SOFT to calculate flux surfaces
        """
        # Find plasma edge
        r0, rmax = self.getPlasmaBoundaries()

        # Convert from normalized to absolute
        if normalizedRadii is not None:
            if radii is not None:
                raise ValueError("Only one of 'normalizedRadii' and 'radii' can be given")

            radii = r0 + normalizedRadii * (rmax-r0)

        # Set default surfaces
        if radii is None:
            radii = r0 + np.array([0.15,0.3,0.45,0.6,0.75,0.9]) * (rmax-r0)

        # Loop over radii, generating and running pi files
        momentum = 1.5e7
        pitchangle = 0.1
        flux = {'R': [], 'Z': [], 'lengths': []}
        for r in radii:
            T, X, Y, Z = self.runParticle(r, momentum, pitchangle, drifts=False)

            flux['R'].append(np.sqrt(X**2 + Y**2))
            flux['Z'].append(Z)
            flux['lengths'].append(len(T))

        flux['R'] = np.array(flux['R'])
        flux['Z'] = np.array(flux['Z'])
        flux['lengths'] = np.array(flux['lengths'])
        self.flux = flux

    def evaluateB(self, r, z):
        """
        fBr = scipy.interpolate.interp2d(self._meshR, self._meshZ, self.B[0], kind='linear')
        fBphi = scipy.interpolate.interp2d(self._meshR, self._meshZ, self.B[1], kind='linear')
        fBz = scipy.interpolate.interp2d(self._meshR, self._meshZ, self.B[2], kind='linear')

        Br = fBr(r, z)[0]
        Bphi = fBphi(r, z)[0]
        Bz = fBz(r, z)[0]
        """
        R = self._meshR.reshape((self._meshR.size,))
        Z = self._meshZ.reshape((self._meshZ.size,))

        rBr = self.B[0].reshape((self.B[0].size,))
        rBphi=self.B[1].reshape((self.B[1].size,))
        rBz = self.B[2].reshape((self.B[2].size,))

        Br = scipy.interpolate.griddata((R,Z), rBr, ([r],[z]), method='linear')
        Bphi=scipy.interpolate.griddata((R,Z),rBphi,([r],[z]), method='linear')
        Bz = scipy.interpolate.griddata((R,Z), rBz, ([r],[z]), method='linear')

        B = np.sqrt(Br**2 + Bphi**2 + Bz**2)
        return Br, Bphi, Bz, B

    def loadDataFile(self, filename):
        """
        Load a file containing a SOFT magnetic equilibrium
        """
        R, Z = 0, 0
        self._meqfile = filename

        if filename.endswith('.mat'):
            # Try to load old MAT-format file. Else, load
            # it as new (HDF5) MAT-format file.
            try:
                matfile = scipy.io.loadmat(filename)

                self.B = (np.array(matfile['Br']), np.array(matfile['Bphi']), np.array(matfile['Bz']))
                self.description = matfile['desc'][0]
                self.maxis = matfile['maxis'][0,:]
                self.name = matfile['name'][0]
                R = matfile['r'][0,:]
                self.separatrix = matfile['separatrix']
                self.wall = matfile['wall']
                Z = matfile['z'][0,:]

                self._inputIsHDF5 = False
            except NotImplementedError:
                matfile = h5py.File(filename)

                self.B = (matfile['Br'][:,:], matfile['Bphi'][:,:], matfile['Bz'][:,:])
                self.description = matfile['desc'][0]
                self.maxis = matfile['maxis'][0,:]
                self.name = matfile['name'][0]
                R = matfile['r'][0,:]
                self.separatrix = matfile['separatrix'][:,:]
                self.wall = matfile['wall'][:,:]
                Z = matfile['z'][0,:]

                self._inputIsHDF5 = True
        elif filename.endswith('h5') or filename.endswith('hdf5'):
            matfile = h5py.File(filename)

            self.B = (matfile['Br'][:,:], matfile['Bphi'][:,:], matfile['Bz'][:,:])
            self.description = matfile['desc']
            self.maxis = matfile['maxis'][0,:]
            self.name = matfile['name']
            R = matfile['r'][0,:]
            self.separatrix = matfile['separatrix'][0,:]
            self.wall = matfile['wall'][0,:]
            Z = matfile['z'][0,:]

            self._inputIsHDF5 = True
        else:
            raise NotImplementedError('Unrecognized data format of file: %s' % filename)

        self.setRZ(R, Z)

    def runParticle(self, radius, momentum, pitchangle, particleOrbit=False, drifts=True, gc_position=True):
        """
        Simulates the given particle orbit (for an electron) with SOFT. If
        'particleOrbit' is True, then the full particle orbit is followed. The
        default (False) means that the guiding-center orbit is followed.
        """

        # First, we simulate the GC orbit
        pi = self.runParticle_pi(radius, momentum, pitchangle, gc_position=(gc_position and not particleOrbit), drifts=(drifts or particleOrbit))
        SIM = self.runSOFT(pi)

        T, X, Y, Z = SIM[:,0], SIM[:,1], SIM[:,2], SIM[:,3]

        # Simulate particle orbit
        if particleOrbit:
            pi = self.runParticle_pi(radius, momentum, pitchangle, particleOrbit=particleOrbit, time=T[-1])
            SIM = self.runSOFT(pi)

            T, X, Y, Z = SIM[:,0], SIM[:,1], SIM[:,2], SIM[:,3]

        return T, X, Y, Z

    def runParticle_pi(self, radius, momentum, pitchangle, particleOrbit=False, gc_position=False, drifts=True, time=-1.0):
        """
        radius: Radius at which to initialize particle
        momentum: Momentum with which to initialize particle
        pitchangle: Pitch angle with which to initialize particle
        particleOrbit: True = Follow full particle orbit, False = Follow guiding-center orbit
        gc_position: For GC orbit: The radius specifies GC position, not particle position
        drifts: For GC orbit: Include drift terms in GC equations of motion
        time: End time of the simulation (note: negative times only make sense for GC orbits)
        """
        pi = ""

        if particleOrbit:
            pi += "useequation=particle-relativistic;\n"
        else:
            pi += "useequation=guiding-center-relativistic;\n"

        pi += "usetool=orbit;\n"
        if drifts: pi += "nodrifts=no;\n"
        else: pi += "nodrifts=yes;\n"

        pi += "magnetic_field=numeric;\n"
        pi += "magnetic numeric { file="+self._meqfile+"; }\n"

        pi += "particles {\n"

        if gc_position: pi += "    gc_position=yes;\n"
        else: pi += "    gc_position=no;\n"

        pi += "    t=0,"+str(time)+";\n"

        pi += "    r="+str(radius)+","+str(radius)+",1;\n"
        pi += "    p="+str(momentum)+","+str(momentum)+",1;\n"
        pi += "    pitch="+str(pitchangle)+","+str(pitchangle)+",1;\n"
        pi += "}\n"

        pi += "tool orbit {\n"
        pi += "    output=@stderr;\n"
        pi += "}\n"

        return pi

    def savePlot(self, filename):
        # TODO Work in progress...
        self.axes.set_axis_off()
        self.figure.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.axes.get_xaxis().set_major_locator(matplotlib.ticker.NullLocator())
        self.axes.get_yaxis().set_major_locator(matplotlib.ticker.NullLocator())

        self.canvas.print_figure(filename, bbox_inches='tight', pad_inches=0, facecolor='black')

    def update(self):
        self.canvas.draw()

    #TODO TODO TODO
    def updateNameAndDescription(self, name, desc):
        """
        Update the name and description fields of the eq file
        """
        if self._inputIsHDF5:
            pass
        else:
            pass

    ####################################################
    #
    # SEMI-PUBLIC PLOT ROUTINES
    #
    ####################################################
    def clearPlot(self):
        """
        Clear the canvas
        """
        self.figure.clear()

        # Reset plot handles
        self._fluxOverlayHandles = []
        self._magneticAxisHandle = None
        self._orbitHandles = []
        self._separatrixOverlayHandle = None
        self._wallOverlayHandle = None

    def plotOverlays(self):
        """
        Plot wall/flux surface overlays as specified in
        the 'overlays' list.
        """
        if self.overlayFluxSurfaces:
            self.plotFluxSurfaces()
        if self.overlayMagneticAxis:
            self.plotMagneticAxis()
        if self.overlaySeparatrix:
            self.plotSeparatrix()
        if self.overlayWallCrossSection:
            self.plotWallCrossSection()

    def plotEq(self):
        """
        Plot the magnetic field
        """
        Bsum = 0

        if self.plotBr:   Bsum += self.B[0]**2
        if self.plotBphi: Bsum += self.B[1]**2
        if self.plotBz:   Bsum += self.B[2]**2

        B = np.sqrt(Bsum)
        if not hasattr(B, "__len__"): return

        self.axes.contour(self._meshR, self._meshZ, B)

        self.axes.axis('equal')

    def plotFluxSurfaces(self, plotstyle='k-', linewidth=2):
        """
        Overlay the plot with flux surfaces
        """
        if self.flux is None:
            raise ValueError('No flux surfaces have been provided!')

        self.removeFluxSurfaces()

        R = self.flux['R']
        Z = self.flux['Z']
        lengths = self.flux['lengths']
        for i in range(0, len(lengths)):
            h = self.axes.plot(R[i][:lengths[i]], Z[i][:lengths[i]], plotstyle, linewidth=linewidth)
            self._fluxOverlayHandles.append(h.pop(0))

    def removeFluxSurfaces(self):
        """
        Remove all painted flux surfaces (if any)
        """
        if self._fluxOverlayHandles is not None:
            for h in self._fluxOverlayHandles:
                h.remove()

            self._fluxOverlayHandles = []
        self.overlayFluxSurfaces = False

    def plotMagneticAxis(self, plotstyle='rs', linewidth=3):
        """
        Plot the magnetic axis
        """

        self.removeMagneticAxis()
        l = self.axes.plot(self.maxis[0], self.maxis[1], plotstyle, linewidth=linewidth)
        self._magneticAxisHandle = l.pop(0)
        self.overlayMagneticAxis = True

    def removeMagneticAxis(self):
        """
        Removes the magnetic axis overlay from the plot
        """

        if self._magneticAxisHandle is not None:
            self._magneticAxisHandle.remove()
            self._magneticAxisHandle = None

        self.overlayMagneticAxis = False

    def plotWallCrossSection(self, plotstyle='k', linewidth=3):
        """
        Paint the wall cross section
        """

        self.removeWallCrossSection()
        R = self.wall[0,:]
        Z = self.wall[1,:]
        l = self.axes.plot(R, Z, plotstyle, linewidth=linewidth)
        self._wallCrossSectionOverlayHandle = l.pop(0)
        self.overlayWallCrossSection = True

    def removeWallCrossSection(self):
        """
        Removes any wall cross section overlay plotted
        over the image.
        """
        if self._wallCrossSectionOverlayHandle is not None:
            self._wallCrossSectionOverlayHandle.remove()
            self._wallCrossSectionOverlayHandle = None

        self.overlayWallCrossSection = False

    def plotSeparatrix(self, plotstyle='r', linewidth=2):
        """
        Plots a separatrix ovelay over the image.
        Also toggles the setting so that 'assembleImage' will
        automatically include the overlay.
        """
        if self.separatrix is None:
            raise ValueError("No separatrix data has been provided!")

        self.removeSeparatrix()
        R = self.separatrix[0,:]
        Z = self.separatrix[1,:]
        self._separatrixOverlayHandle = self.axes.plot(R, Z, plotstyle, linewidth=linewidth)
        self.overlaySeparatrix = True

    def removeSeparatrix(self):
        """
        Removes any separatrix overlay imposed over the image
        """
        if self._separatrixOverlayHandle is not None:
            self._separatrixOverlayHandle.remove()
            self._separatrixOverlayHandle = None

        self.overlaySeparatrix = False

    def plotOrbit(self, R, Z, plotstyle=None, linewidth=1):
        """
        Plots the orbit specified by R and Z coordinates
        """

        if plotstyle is None:
            plotstyle = self.getNextOrbitStyle()

        l = self.axes.plot(R, Z, plotstyle, linewidth)
        self._orbitHandles.append(l.pop(0))

    def clearOrbits(self):
        """
        Remove all orbits from the plot
        """
        if self._orbitHandles is not None:
            for h in self._orbitHandles:
                h.remove()

            self._orbitHandles = []

    def getNextOrbitStyle(self):
        """
        Get the linestyle to use for next orbit
        """
        clrs = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        styles = [s for s in clrs] + [s+'--' for s in clrs] + [s+':' for s in clrs] + [s+'-.' for s in clrs]

        i = len(self._orbitHandles) % len(styles)

        return styles[i]

    def runSOFT(self, pifile):
        """
        Run SOFT, passing the given pifile on stdin.
        The contents of stderr are interpreted as CSV output
        and are returned as an array.
        """
        if self.SOFTPATH is None:
            raise RuntimeError('The path to SOFT has not been specified')

        p = subprocess.Popen([self.SOFTPATH+'/soft'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        stderr_data = p.communicate(input=bytearray(pifile, 'ascii'))[1].decode('utf-8')

        if p.returncode is not 0:
            raise RuntimeError('SOFT exited with a non-zero exit code.')

        c = io.StringIO(stderr_data)
        return np.loadtxt(c, delimiter=',', skiprows=1)

    ####################################################
    #
    # STATIC METHODS
    #
    ####################################################
    @staticmethod
    def registerGeriMap():
        """
        Register the perceptually uniform colormap 'GeriMap' with matplotlib
        """
        gm = [(0, 0, 0), (.15, .15, .5), (.3, .15, .75),
              (.6, .2, .50), (1, .25, .15), (.9, .5, 0),
              (.9, .75, .1), (.9, .9, .5), (1, 1, 1)]
        gerimap = LinearSegmentedColormap.from_list('GeriMap', gm)
        gerimap_r = LinearSegmentedColormap.from_list('GeriMap_r', gm[::-1])
        plt.register_cmap(cmap=gerimap)
        plt.register_cmap(cmap=gerimap_r)

