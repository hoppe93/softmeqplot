from PyQt5 import QtWidgets
from ui import main_design
from PlotWindow import PlotWindow
import sys
import os.path
import numpy as np
import scipy.io
import h5py
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox
from MeqPlot import MeqPlot


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = main_design.Ui_MainWindow()
        self.ui.setupUi(self)

        # Create plot window
        self.plotWindow = PlotWindow()
        self.meqplot = MeqPlot(self.plotWindow.figure, self.plotWindow.canvas)

        # Bind to events
        self.bindEvents()

        # Check command-line arguments
        if len(sys.argv) == 2:
            if os.path.isfile(sys.argv[1]):
                self.ui.tbMeqFile.setText(os.path.abspath(sys.argv[1]))
                self.loadFile(sys.argv[1])

    def bindEvents(self):
        # Browse
        self.ui.btnBrowse.clicked.connect(self.openFile)

        # Metadata (TODO)
        self.ui.btnSaveMetadata.clicked.connect(self.saveMetadata)

        # Plot
        self.ui.cbBr.stateChanged.connect(self.cbPlotChanged)
        self.ui.cbBphi.stateChanged.connect(self.cbPlotChanged)
        self.ui.cbBz.stateChanged.connect(self.cbPlotChanged)
        self.ui.cbWall.stateChanged.connect(self.cbPlotChanged)
        self.ui.cbSeparatrix.stateChanged.connect(self.cbPlotChanged)
        self.ui.cbFlux.stateChanged.connect(self.cbPlotChanged)
        self.ui.cbMaxis.stateChanged.connect(self.cbPlotChanged)

        # Orbit
        self.ui.btnGCOrbit.clicked.connect(self.plotGCOrbit)
        self.ui.btnParticleOrbit.clicked.connect(self.plotParticleOrbit)
        self.ui.btnClearOrbits.clicked.connect(self.clearOrbits)

        # Plot-window
        self.plotWindow.canvas.mpl_connect('button_press_event', self.pointSelected)

    def cbPlotChanged(self):
        self.meqplot.plotBr = self.ui.cbBr.isChecked()
        self.meqplot.plotBphi = self.ui.cbBphi.isChecked()
        self.meqplot.plotBz = self.ui.cbBz.isChecked()
        self.meqplot.overlayFluxSurfaces = self.ui.cbFlux.isChecked()
        self.meqplot.overlaySeparatrix = self.ui.cbSeparatrix.isChecked()
        self.meqplot.overlayWallCrossSection = self.ui.cbWall.isChecked()
        self.meqplot.overlayMagneticAxis = self.ui.cbMaxis.isChecked()

        self.refreshImage()

    def closeEvent(self, event):
        self.exit()

    def exit(self):
        self.plotWindow.close()
        self.close()

    def loadFile(self, filename):
        self.ui.tbMeqFile.setText(filename)
        self.filename = filename
        self.meqplot.loadDataFile(filename)

        self.meqfileUpdated()

        if not self.meqplot.hasSeparatrix():
            self.ui.cbSeparatrix.setChecked(False)
            self.ui.cbSeparatrix.setEnabled(False)

        # Enable things to plot
        self.cbPlotChanged()

    def meqfileUpdated(self):
        if self.filename is not None:
            self.ui.gbMetadata.setEnabled(True)
            self.ui.gbPlot.setEnabled(True)
            self.ui.gbOrbits.setEnabled(True)
        else:
            self.ui.gbMetadata.setEnabled(False)
            self.ui.gbPlot.setEnabled(False)
            self.ui.gbOrbits.setEnabled(False)

        self.ui.tbName.setText(self.meqplot.name)
        self.ui.tbDescription.setPlainText(self.meqplot.description)
        self.ui.lblMaxis.setText('(%.3f, %.3f)' % (self.meqplot.maxis[0], self.meqplot.maxis[1]))

        # Calculate flux surfaces for future use
        self.meqplot.calculateFluxSurfaces()

        # Get plasma boundaries
        r0, rmax = self.meqplot.getPlasmaBoundaries()

        self.ui.dsbRadius.setRange(r0, rmax)
        self.ui.dsbRadius.setValue(r0+(rmax-r0)*0.6)

        # Calculate magnetic field at axis
        self.evalBAt(self.meqplot.maxis[0], self.meqplot.maxis[1])

    def openFile(self):
        filename, _ = QFileDialog.getOpenFileName(parent=self, caption="Open SOFT Magnetic Equilibrium file", filter="SOFT Magnetic Equilibrium (*.mat);;All files (*.*)")

        if filename:
            self.loadFile(filename)

    def refreshImage(self):
        if not self.plotWindow.isVisible():
            self.plotWindow.show()

        #self.plotWindow.plotImage(self.meqplot)
        self.meqplot.assemblePlot()
        self.meqplot.axes.axis('equal')
        self.plotWindow.drawSafe()

    def saveMetadata(self):
        self.meqplot.updateNameAndDescription(self.ui.tbName.value(), self.ui.tbDescription.value())

    #############################
    #
    # ORBITS
    #
    #############################
    def clearOrbits(self):
        self.meqplot.clearOrbits()
        self.meqplot.update()

    def plotGCOrbit(self):
        r = self.ui.dsbRadius.value()
        p = self.ui.dsbMomentum.value() * 1e6
        theta = self.ui.dsbPitch.value()

        try:
            T, X, Y, Z = self.meqplot.runParticle(r, p, theta, gc_position=False)
            R = np.sqrt(X**2 + Y**2)

            self.meqplot.plotOrbit(R, Z)
            self.meqplot.update()
        except RuntimeError as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(str(e))
            msg.setWindowTitle('SOFT Error')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def plotParticleOrbit(self):
        r = self.ui.dsbRadius.value()
        p = self.ui.dsbMomentum.value() * 1e6
        theta = self.ui.dsbPitch.value()

        try:
            T, X, Y, Z = self.meqplot.runParticle(r, p, theta, particleOrbit=True)
            R = np.sqrt(X**2 + Y**2)

            self.meqplot.plotOrbit(R, Z)
            self.meqplot.update()
        except RuntimeError as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(str(e))
            msg.setWindowTitle('SOFT Error')
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def evalBAt(self, R, Z):
        Br, Bphi, Bz, B = self.meqplot.evaluateB(R, Z)
    
        self.ui.lblSampledB.setText("(%.3f, %.3f) m" % (R,Z))
        self.ui.lblBStrength.setText("%.3f T" % B)
        self.ui.lblBComp.setText("(%.3f, %.3f, %.3f) T" % (Br, Bphi, Bz))

    def pointSelected(self, event):
        R = event.xdata
        Z = event.ydata

        self.evalBAt(R, Z)

