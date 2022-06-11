import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import SourceWindow, TargetWindow, MeanValueSeamlessCloning
from tkinter import filedialog
from tkinter import messagebox
import cv2
import numpy as np

window = None

class MainWindow:
    _instance = None
    @staticmethod
    def get_instance():
        if MainWindow._instance is None:
            MainWindow()
        return MainWindow._instance

    def __init__(self):
        if MainWindow._instance is not None:
            raise Exception('only one instance can exist')
        else:
            MainWindow._instance = self
            self.window = tk.Tk()
            self.window.title('InstantImageCloning')
            self.window.geometry( '600x500' )

            self.sourceImg = Image.open( 'pic/source.png' )
            self.sourceShow = self.sourceImg.copy()
            self.sourcemask = None
            self.sourceBoundaryVertex = list()  # local coord

            self.targetImg = Image.open( 'pic/target.png' )
            self.targetShow = self.targetImg.copy()
            self.centerCoord = None

            self.resultImg = Image.open( 'pic/result.png' )
            self.resultShow = self.resultImg.copy()

            self.CreateButton()
            
            self.SetImg( 'source', self.sourceImg )
            self.SetImg( 'target', self.targetImg)
            self.SetImg( 'result', self.resultImg )

            self.sourceWindow = SourceWindow.init( MainWindow._instance )
            self.targetWindow = TargetWindow.init( MainWindow._instance )
            
    def MainLoop(self):
        self.window.mainloop()

    def SetImg( self, imgTarget, img ):
        if imgTarget == 'source':
            self.sourceShow = img.resize((300, 200))
            imgTK = ImageTk.PhotoImage( self.sourceShow )
            self.sourceLabel = tk.Label( self.window , width = 300, height = 200, image = imgTK )
            self.sourceLabel.image = imgTK
            self.sourceLabel.grid( row = 1 , column = 0 )

        elif imgTarget == 'target':
            self.targetShow = img.resize((300, 200))
            imgTK = ImageTk.PhotoImage( self.targetShow )
            self.targetLabel = tk.Label( self.window , width = 300, height = 200, image = imgTK )
            self.targetLabel.image = imgTK
            self.targetLabel.grid( row = 3 , column = 0 )

        elif imgTarget == 'result':
            self.resultShow = img.resize((300, 200))
            imgTK = ImageTk.PhotoImage( self.resultShow )
            self.resultLabel = tk.Label( self.window , width = 300, height = 200, image = imgTK )
            self.resultLabel.image = imgTK
            self.resultLabel.grid( row = 1 , column = 1 )

    def CreateButton( self ):

        self.sourceButton = tk.Button(self.window, command = self.SourceClick , text='open source image', bg='blue', fg='white', font=('Arial', 12) )
        self.sourceButton.grid( row = 0, column = 0 )

        self.targetButton = tk.Button(self.window , command = self.TargetClick , text='open target image', bg='red', fg='white', font=('Arial', 12) )
        self.targetButton.grid( row = 2, column = 0 )

        self.resultButton = tk.Button(self.window , command = self.BlendingClick , text='start blending', bg='green', fg='white', font=('Arial', 12) )
        self.resultButton.grid( row = 0, column = 1 )

        self.resultButton = tk.Button(self.window , command = self.SaveClick , text='save result', bg='green', fg='white', font=('Arial', 12) )
        self.resultButton.grid( row = 2, column = 1 )

    def SourceClick( self ):
        print('source button clicked')
        self.sourceImg = self.OpenImage()
        self.sourceShow = self.sourceImg.copy()
        self.sourceWindow = SourceWindow.init( MainWindow._instance )
        self.sourceWindow.SetImg( self.sourceImg )
        self.sourceWindow.UpdateImg()
        self.sourceWindow.MainLoop()

    def UpdateSource( self , img , mask , boundaryVertex ):
        print('update source')
        img = Image.fromarray( img )
        self.sourcemask = mask
        self.sourceBoundaryVertex = boundaryVertex
        self.SetImg( 'source', img )

    def UpdateTarget( self, img , centerCoord ):
        img = Image.fromarray( img )
        self.centerCoord = centerCoord
        self.SetImg( 'target', img )

    def TargetClick( self ):
        print('target button clicked')
        if self.sourcemask is not None:
            self.targetImg = self.OpenImage()
            self.targetShow = self.targetImg.copy()
            self.targetWindow = TargetWindow.init( MainWindow._instance )
            self.targetWindow.SetImg( self.targetImg )
            self.targetWindow.SetBoundaryVertex( self.sourceBoundaryVertex )
            self.targetWindow.UpdateImg()
            self.targetWindow.MainLoop()
        else:
            messagebox.showinfo( 'Hint','請先Crop Source Image後再處理 Target Image')
    
    def BlendingClick( self ):
        print('Blending button clicked')
        self.resultShow = MeanValueSeamlessCloning.Start( self.sourceImg ,  self.sourcemask , self.sourceBoundaryVertex , self.targetImg , self.centerCoord )
        self.SetImg( 'result' , self.resultShow )

    def SaveClick( self ):
        print('Save button clicked')
        tmp = np.array( self.resultShow )
        tmp = cv2.cvtColor( tmp , cv2.COLOR_RGB2BGR )
        cv2.imwrite( 'result.jpg' , tmp )
        print('Save Done')

    def OpenImage(self):
        filePath = filedialog.askopenfilename(
            filetypes =
                [
                    ('imgae files',('.jpeg', '.jpg'))
                ]
            )
        self.imgPath = filePath
        img = Image.open( self.imgPath )
        return img

def GetMainWindow():
    window = MainWindow().get_instance()
    return window