import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import SourceImg
from tkinter import filedialog

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

            self.sourceImg = Image.open( 'source.png' )
            self.targetImg = Image.open( 'target.png' )
            self.resultImg = Image.open( 'result.png' )
            self.CreateButton()
            
            self.SetImg( 'source', self.sourceImg )
            self.SetImg( 'target', self.targetImg)
            self.SetImg( 'result', self.resultImg )

            self.sourceWindow = None
            self.targetWindow = None
            self.resultWindow = None
            
    def MainLoop(self):
        self.window.mainloop()

    def SetImg( self, imgTarget, img ):
        if imgTarget == 'source':
            self.sourceImg = img.resize((300, 200))
            imgTK = ImageTk.PhotoImage( self.sourceImg )
            self.sourceLabel = tk.Label( self.window , width = 300, height = 200, image = imgTK )
            self.sourceLabel.image = imgTK
            self.sourceLabel.grid( row = 1 , column = 0 )

        elif imgTarget == 'target':
            self.targetImg = img.resize((300, 200))
            imgTK = ImageTk.PhotoImage( self.targetImg )
            self.targetLabel = tk.Label( self.window , width = 300, height = 200, image = imgTK )
            self.targetLabel.image = imgTK
            self.targetLabel.grid( row = 3 , column = 0 )

        elif imgTarget == 'result':
            self.resultImg = img.resize((300, 200))
            imgTK = ImageTk.PhotoImage( self.resultImg )
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
        self.sourceWindow = SourceImg.GetInstance( self.sourceImg, MainWindow._instance )
        self.sourceWindow.MainLoop()

    def UpdateSourceImg( self , img ):
        print('update source')
        img = Image.fromarray( img )
        self.SetImg( 'source', img )

    def TargetClick( self ):
        print('target button clicked')
    
    def BlendingClick( self ):
        print('Blending button clicked')

    def SaveClick( self ):
        print('Save button clicked')
    
    

    def OpenImage(self):
        filePath = filedialog.askopenfilename(
            filetypes =
                [
                    ('imgae files',('.png', '.jpg'))
                ]
            )
        self.imgPath = filePath
        img = Image.open( self.imgPath )
        return img

def GetMainWindow():
    window = MainWindow().get_instance()
    return window