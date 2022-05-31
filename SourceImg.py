import tkinter as tk
import numpy as np
from turtle import width
from venv import create
from tkinter import filedialog
from tkinter import *
import cv2
from PIL import Image, ImageTk, ImageDraw
from tkinter import messagebox

class SourceImgWindow:
    _instance = None
    @staticmethod
    def get_instance():
        if SourceImgWindow._instance is None:
            SourceImgWindow()
        return SourceImgWindow._instance

    def __init__( self ):
        if SourceImgWindow._instance is not None:
            raise Exception('only one instance can exist')
        else:
            SourceImgWindow._instance = self
            self.window = tk.Tk()
            self.window.title('SourceImgEditor')
            self.window.bind('<Button-3>', self.ClickAddVertex )
            self.window.bind('<Button-2>', self.ClickDeleteVertex )
            self.window.bind('<Button-1>', self.ClickDrawLine )
            self.boundaryVertex = list()

    def OpenImage(self):
        self.window.withdraw()
        filePath = filedialog.askopenfilename(
            filetypes =
                [
                    ('imgae files',('.png', '.jpg'))
                ]
            )
        self.imgPath = filePath
        self.originImg = Image.open( self.imgPath )
        self.modifiedImg = self.originImg.copy()
        self.draw = ImageDraw.Draw( self.modifiedImg )
        self.SetImg()

    def SetImg(self):
        self.imgTK = ImageTk.PhotoImage( self.modifiedImg )
        self.label = tk.Label( self.window, image = self.imgTK )
        self.label.image = self.imgTK
        self.label.grid( column = 0, row = 0 )
        self.window.geometry( str(self.originImg.width) + 'x' + str(self.originImg.height) )

    def RefreshImg(self):
        self.imgTK = ImageTk.PhotoImage( self.originImg )
        self.label = tk.Label( self.window, image = self.imgTK )
        self.label.image = self.imgTK
        self.label.grid( column = 0, row = 0 )
        self.window.geometry( str(self.originImg.width) + 'x' + str(self.originImg.height) )

    def MainLoop(self):
        self.window.deiconify()
        self.window.mainloop()

    def ResetImg(self):
        self.modifiedImg = self.originImg.copy()
        self.draw = ImageDraw.Draw( self.modifiedImg )

    def PushBoundaryVertex(self, x, y):
        if( x >= 0 and x < self.originImg.width and y >= 0 and y< self.originImg.height ):
            self.boundaryVertex.append( (x,y) )
            print(self.boundaryVertex)
            self.draw.ellipse( (x-3,y-3,x+3,y+3), fill = 'blue', outline = 'blue' )
            self.SetImg()
        else:
            self.OutOfBoundWarning( x, y )

    def OutOfBoundWarning( self,x, y):
        messagebox.showinfo( 'Warning','Out of Bound: {} {} '.format(x,y))

    def PopBoundaryVertex( self ):
        if len(self.boundaryVertex) != 0 :
            coord = self.boundaryVertex.pop( len( self.boundaryVertex ) - 1 )
            x , y = coord
            pixels = self.modifiedImg.load()
            originPixels = self.originImg.load()
            for i in range( x-5, x+5 ):
                for j in range( y-5, y+5):
                    if( x >= 0 and x < self.originImg.width and y >= 0 and y< self.originImg.height ):
                        pixels[i,j] = originPixels[i,j]
            self.DrawLine()

    def DrawLine( self ):
        if len( self.boundaryVertex ) > 1:
            self.ResetImg()
            for i in range( len(self.boundaryVertex) ):
                x , y = self.boundaryVertex[ i ]
                self.draw.ellipse( (x-3,y-3,x+3,y+3), fill = 'blue', outline = 'blue' )
            self.draw.line( self.boundaryVertex, fill = 'red', width = 3 )
            self.draw.line( [ self.boundaryVertex[0], self.boundaryVertex[ len(self.boundaryVertex) - 1 ] ], fill = 'red', width = 3 )
        elif len(self.boundaryVertex) == 1:
            self.ResetImg()
            x , y = self.boundaryVertex[ 0 ]
            self.draw.ellipse( (x-3,y-3,x+3,y+3), fill = 'blue', outline = 'blue' )
        self.SetImg()

    def CropImg( self ):
        # 超過三個點才做
        if len( self.boundaryVertex ) > 2:
            imgNumpy = np.asarray( self.originImg )
            mask_img = Image.new( '1', ( imgNumpy.shape[1] , imgNumpy.shape[0] ) , 0 )
            ImageDraw.Draw( mask_img ).polygon( self.boundaryVertex , outline = 1 , fill = 1 )
            self.mask = np.array( mask_img )
            self.cropImg = np.empty( imgNumpy.shape, dtype = 'uint8' )
            self.cropImg = imgNumpy
            
            self.cropImg[:,:,0] = self.cropImg[:,:,0] * self.mask
            self.cropImg[:,:,1] = self.cropImg[:,:,1] * self.mask
            self.cropImg[:,:,2] = self.cropImg[:,:,2] * self.mask
            cv2.imshow( 'cropImg' , self.cropImg )
            cv2.waitKey(0)

    # callback
    def ClickAddVertex(self, event ):
        self.PushBoundaryVertex( event.x , event.y )

    def ClickDeleteVertex(self, event ):
        self.PopBoundaryVertex()

    def ClickDrawLine( self, evevt ):
        self.DrawLine()
        self.CropImg()

def init():
    sourceImgWindow = SourceImgWindow()
    return sourceImgWindow
