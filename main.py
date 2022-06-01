import cv2
import numpy as np
import SourceWindow
import MainWindow

if __name__=='__main__':
    mainWindow = MainWindow.GetMainWindow()
    mainWindow.get_instance().MainLoop()
    # sourceImgWindow = SourceImg.init()
    # sourceImgWindow.get_instance().OpenImage()
    # sourceImgWindow.get_instance().MainLoop()