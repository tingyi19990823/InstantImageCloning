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

from pip import main
import MainWindow

class TargetWindow( tk.Toplevel ):
    def __init__( self, mainWindowInstance ):
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.onClosing)
        self.bind('<Button-3>', self.ClickRight )
        self.originImg = None
        self.modifiedImg = None
        self.mainWindow = mainWindowInstance
        self.centerCoord = None
        self.originCenterCoord = None
        self.draw = None
        self.scale = 1

        # source info
        self.boundaryVertex = None
        # 回傳給 Main Window

        self.CreateButton()

    def onClosing( self ):
        self.withdraw()

    def ClickRight( self , event ):
        if( event.x >= 0 and event.x < self.modifiedImg.width and event.y >= 0 and event.y < self.modifiedImg.height ):
            self.setCenterCoord( event.x, event.y )
        else:
            messagebox.showinfo( 'Warning','Out of Bound: {} {} '.format( event.x , event.y ) )

    def setCenterCoord( self, x , y ):
        if self.centerCoord is not None:
            modifiedPixels = self.modifiedImg.load()
            originPixels = self.originImg.load()
            oldX , oldY = self.centerCoord

            for i in range( oldX - 5 , oldX + 5 ):
                for j in range( oldY - 5 , oldY + 5):
                    if( i >= 0 and i < self.modifiedImg.width and j >= 0 and j < self.modifiedImg.height ):
                        originX = int( i / self.scale )
                        originY = int( j / self.scale )
                        modifiedPixels[ i , j ] = originPixels[ originX , originY ]
        self.centerCoord = ( x , y )
        self.originCenterCoord = ( int( x / self.scale ) , int( y / self.scale ) )
        self.draw.ellipse( (x-3,y-3,x+3,y+3), fill = 'blue', outline = 'white' )
        self.UpdateImg()

    def CreateButton(self):
        self.cropButton = tk.Button( self , command = self.WrapImg , text='Wrap', bg='blue', fg='white', font=('Arial', 12) ).grid(row = 1, column = 0)
        self.doneButton =  tk.Button( self , command = self.DoneFunc , text='Done', bg='red', fg='white', font=('Arial', 12) ).grid(row = 2, column = 0)

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

    def SetBoundaryVertex( self, boundaryVertex ):
        offsetRow = 0
        offsetCol = 0
        length = len( boundaryVertex )
        for i in range( length ):
            offsetCol += boundaryVertex[ i ][ 0 ] / length
            offsetRow += boundaryVertex[ i ][ 1 ] / length
        localCoord = boundaryVertex.copy()
        for i in range( length ):
            x = localCoord[ i ][ 0 ] - int(offsetCol)
            y = localCoord[ i ][ 1 ] - int(offsetRow)
            localCoord[ i ] = ( x , y )
        self.boundaryVertex = localCoord

    def UpdateImg( self ):
        self.imgTK = ImageTk.PhotoImage( self.modifiedImg )
        self.label = tk.Label( self, image = self.imgTK )
        self.label.image = self.imgTK
        self.label.grid( column = 0, row = 0 )

    def WrapImg( self ):
        worldCoord = list()
        if self.centerCoord is not None:
            for i in range( len( self.boundaryVertex ) ):
                x , y = self.boundaryVertex[ i ]
                x = int( x * self.scale )
                y = int( y * self.scale )
                x += self.centerCoord[ 0 ]
                y += self.centerCoord[ 1 ]
                worldCoord.append( ( x , y ) )
                if( x < 0 or x >= self.modifiedImg.width or y < 0 or y >= self.modifiedImg.height ):
                    messagebox.showinfo( 'Warning',' out of bound ')
                    return
            self.draw.polygon( worldCoord , outline = 'red' )
        self.UpdateImg()
    
    def DoneFunc( self ):
        img = self.originImg
        draw = ImageDraw.Draw( img )
        worldCoord = []

        if self.originCenterCoord is not None:
            for i in range( len( self.boundaryVertex ) ):
                x , y = self.boundaryVertex[ i ]
                x += self.originCenterCoord[ 0 ]
                y += self.originCenterCoord[ 1 ]
                worldCoord.append( ( x , y ) )
            draw.polygon( worldCoord , outline = 'red' )
            img = np.array( img )
            self.mainWindow.UpdateTarget( img , self.originCenterCoord )
            self.withdraw()

    def MainLoop( self ):
        self.deiconify()
        self.mainloop()

def init( mainWindowInstance ):
    window = TargetWindow( mainWindowInstance )
    window.withdraw()
    return window
