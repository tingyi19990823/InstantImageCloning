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
        self.bind('<Button-3>', self.ClickRight )
        self.originImg = None
        self.modifiedImg = None
        self.mainWindow = mainWindowInstance
        self.centerCoord = None

        self.CreateButton()

    def ClickRight( self , event ):
        if( event.x >= 0 and event.x < self.originImg.width and event.y >= 0 and event.y < self.originImg.height ):
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
                    if( i >= 0 and i < self.originImg.width and j >= 0 and j < self.originImg.height ):
                        modifiedPixels[ i , j ] = originPixels[ i , j ]
        self.centerCoord = ( x , y )
        self.draw.ellipse( (x-3,y-3,x+3,y+3), fill = 'blue', outline = 'white' )
        self.UpdateImg()

    def CreateButton(self):
        self.cropButton = tk.Button( self , command = self.WrapImg , text='Wrap', bg='blue', fg='white', font=('Arial', 12) ).grid(row = 1, column = 0)
        self.doneButton =  tk.Button( self , command = self.DoneFunc , text='Done', bg='red', fg='white', font=('Arial', 12) ).grid(row = 2, column = 0)

    def SetImg( self , img ):
        self.originImg = img
        self.modifiedImg = img.copy()
        self.draw = ImageDraw.Draw( self.modifiedImg )
        width = self.originImg.width
        height = self.originImg.height + 64
        self.geometry( str(width) + 'x' + str(height) )

    def UpdateImg( self ):
        self.imgTK = ImageTk.PhotoImage( self.modifiedImg )
        self.label = tk.Label( self, image = self.imgTK )
        self.label.image = self.imgTK
        self.label.grid( column = 0, row = 0 )

    def WrapImg( self , ):
        print('wrap')
    
    def DoneFunc( self ):
        print('done')


def init( mainWindowInstance ):
    window = TargetWindow( mainWindowInstance )
    window.withdraw()
    return window
