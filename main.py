#!/usr/bin/env python3
from PyQt5 import QtWidgets
from MainWindow import MainWindow
import sys

app = None


def show_main():
    global app
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    sys.exit(show_main())
