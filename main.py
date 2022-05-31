import cv2
import numpy as np

import SourceImg
import MainWindow

if __name__=='__main__':
    mainWindow = MainWindow.MainWindow()
    mainWindow.get_instance().MainLoop()
    # sourceImgWindow = SourceImg.init()
    # sourceImgWindow.get_instance().OpenImage()
    # sourceImgWindow.get_instance().MainLoop()