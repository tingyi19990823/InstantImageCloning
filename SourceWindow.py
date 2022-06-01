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

class SourceImgWindow:
    @staticmethod
    def get_instance( img, mainWindowInstance ):
        return SourceImgWindow( img, mainWindowInstance )

    def __init__( self , img , mainWindowInstance ):
        self.mainWindowInstance = mainWindowInstance
        self.window = tk.Toplevel()
        self.window.title('SourceImgEditor')

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.bind('<Button-3>', self.ClickAddVertex )
        self.window.bind('<Button-2>', self.ClickDeleteVertex )
        self.window.bind('<Button-1>', self.ClickDrawLine )
        self.boundaryVertex = list()

        self.originImg = img
        self.modifiedImg = img.copy()
        self.cropImg = None

        self.draw = ImageDraw.Draw( self.modifiedImg )
        self.SetImg()

        self.CreateButton()

    def on_closing(self):
        self.window.destroy()

    def CreateButton(self):
        self.cropButton = tk.Button( self.window, command = self.CropImg , text='Crop', bg='blue', fg='white', font=('Arial', 12) ).grid(row = 1, column = 0)
        self.doneButton =  tk.Button( self.window, command = self.DoneFunc , text='Done', bg='red', fg='white', font=('Arial', 12) ).grid(row = 2, column = 0)

    def SetImg( self ):
        self.imgTK = ImageTk.PhotoImage( self.modifiedImg )
        self.label = tk.Label( self.window, image = self.imgTK )
        self.label.image = self.imgTK
        self.label.grid( column = 0, row = 0 )

        width = self.originImg.width
        height = self.originImg.height + 64
        self.window.geometry( str(width) + 'x' + str(height) )

    def RefreshImg(self):
        self.imgTK = ImageTk.PhotoImage( self.originImg )
        self.label = tk.Label( self.window, image = self.imgTK )
        self.label.image = self.imgTK
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

    def DoneFunc( self ):
        print('done')
        if self.cropImg is not None:
            self.mainWindowInstance.UpdateSourceImg( self.cropImg )
            self.window.destroy()
        else:
            messagebox.showinfo( 'Hint','請先Crop一張圖片再按Done')
    
    # callback
    def ClickAddVertex(self, event ):
        self.PushBoundaryVertex( event.x , event.y )

    def ClickDeleteVertex(self, event ):
        self.PopBoundaryVertex()

    def ClickDrawLine( self, evevt ):
        self.DrawLine()

def GetInstance( img, mainWindowInstance ):
    sourceImgWindow = SourceImgWindow.get_instance( img, mainWindowInstance )
    sourceImgWindow.window.withdraw()
    return sourceImgWindow

