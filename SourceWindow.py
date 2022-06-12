from decimal import localcontext
from genericpath import exists
import tkinter as tk
import numpy as np
from turtle import width
from venv import create
from tkinter import filedialog
from tkinter import *
import cv2
from PIL import Image, ImageTk, ImageDraw
from tkinter import messagebox
import MainWindow

class SourceImgWindow( tk.Toplevel ):
    def __init__( self , mainWindowInstance ):
        super().__init__()
        self.mainWindowInstance = mainWindowInstance
        self.title('SourceImgEditor')

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.bind('<Button-3>', self.ClickAddVertex )
        self.bind('<Button-2>', self.ClickDeleteVertex )
        
        self.originImg = None
        self.modifiedImg = None
        self.draw = None
        self.label = None
        self.scale = 1

        self.CreateButton()
        # 需要回傳給 MainWindow
        self.cropImg = None
        self.mask = None
        self.boundaryVertex = list()
        self.originScaleBoundaryVertex = list()

    def on_closing( self ):
        self.destroy()

    def CreateButton(self):
        self.cropButton = tk.Button( self, command = self.CropImg , text='Crop', bg='blue', fg='white', font=('Arial', 12) ).grid(row = 1, column = 0)
        self.doneButton =  tk.Button( self, command = self.DoneFunc , text='Done', bg='red', fg='white', font=('Arial', 12) ).grid(row = 2, column = 0)

    def SetImg( self , img ):
        self.originImg = img
        self.modifiedImg = img.copy()
        
        width = img.width
        height = img.height

        finalHeight = height
        finalWidth = width
        while( finalWidth > 1000 or finalHeight > 700 ):
            self.scale -= 0.1
            finalWidth = int(width * self.scale)
            finalHeight = int(height * self.scale)

        self.modifiedImg = self.modifiedImg.resize( ( finalWidth , finalHeight ) )
        self.draw = ImageDraw.Draw( self.modifiedImg )

        finalWidth = finalWidth + 10
        finalHeight = finalHeight + 64

        my_geometry = str(finalWidth) + 'x' + str(finalHeight)
        self.geometry( my_geometry )

    def UpdateImg( self ):
        self.imgTK = ImageTk.PhotoImage( self.modifiedImg )
        self.label = tk.Label( self, image = self.imgTK )
        self.label.image = self.imgTK
        self.label.grid( column = 0, row = 0 )

    def MainLoop(self):
        self.deiconify()
        self.mainloop()

    def ResetImg(self):
        self.modifiedImg = self.originImg.copy()
        width = int( self.modifiedImg.width * self.scale )
        height = int( self.modifiedImg.height * self.scale )
        self.modifiedImg = self.modifiedImg.resize( ( width , height ) )

        self.draw = ImageDraw.Draw( self.modifiedImg )

    def PushBoundaryVertex(self, x, y):
        if( x >= 0 and x < self.originImg.width and y >= 0 and y< self.originImg.height ):
            self.boundaryVertex.append( (x,y) )
            self.originScaleBoundaryVertex.append( ( int( x / self.scale ) , int( y / self.scale ) ) )

            print(self.boundaryVertex)

            self.draw.ellipse( (x-3,y-3,x+3,y+3), fill = 'blue', outline = 'white' )

            if len( self.boundaryVertex ) > 1:
                self.DrawLine()
            self.UpdateImg()
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
                    if( i >= 0 and i < self.originImg.width and j >= 0 and j < self.originImg.height ):
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
        self.UpdateImg()

    def CropImg( self ):
        # 超過三個點才做
        if len( self.boundaryVertex ) > 2:
            imgNumpy = np.asarray( self.originImg )
            mask_img = Image.new( '1', ( imgNumpy.shape[1] , imgNumpy.shape[0] ) , 0 )

            ImageDraw.Draw( mask_img ).polygon( self.originScaleBoundaryVertex , outline = 0 , fill = 1 )
            self.mask = np.array( mask_img )
            self.cropImg = np.empty( imgNumpy.shape, dtype = 'uint8' )
            self.cropImg = imgNumpy
            
            self.cropImg[:,:,0] = self.cropImg[:,:,0] * self.mask
            self.cropImg[:,:,1] = self.cropImg[:,:,1] * self.mask
            self.cropImg[:,:,2] = self.cropImg[:,:,2] * self.mask

    def DoneFunc( self ):
        if self.cropImg is not None:
            self.mainWindowInstance.UpdateSource( self.cropImg , self.mask , self.originScaleBoundaryVertex )
            self.destroy()
        else:
            messagebox.showinfo( 'Hint','請先Crop一張圖片再按Done')
    
    # Mouse Callback
    def ClickAddVertex( self, event ):
        self.PushBoundaryVertex( event.x , event.y )

    def ClickDeleteVertex( self, event ):
        self.PopBoundaryVertex()

def init( mainWindowInstance ):
    sourceImgWindow = SourceImgWindow( mainWindowInstance )
    sourceImgWindow.withdraw()
    return sourceImgWindow

