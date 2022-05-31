import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

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
            self.window.geometry( '800x600' )

        self.CreateButton('open source image')
    def MainLoop(self):
        self.window.mainloop()

    def CutFrame(self):
        self.sourceDiv = tk.Frame( self.window, width = 250, height = 250 , bg = 'blue' )
        self.targetDiv = tk.Frame( self.window, width = 250, height = 250 , bg = 'red' )
        self.resultDiv = tk.Frame( self.window, width = 250, height = 500 , bg = 'green' )
        
        self.sourceDiv.grid( column = 0, row = 0 )
        self.targetDiv.grid( column = 0, row = 1 )
        self.resultDiv.grid( column = 1, row = 1 )

    def CreateButton( self, txt ):
        bt_1 = tk.Button(self.window, text=txt, bg='blue', fg='white', font=('Arial', 12))
        bt_1['width'] = 20
        bt_1['height'] = 4
        bt_1['activebackground'] = 'blue'
        bt_1['activeforeground'] = 'white'

        bt_1.grid(column=0, row=0)

    # def SetImg(self):
    #     self.imgTK = ImageTk.PhotoImage( self.modifiedImg )
    #     self.label = tk.Label( self.window, image = self.imgTK )
    #     self.label.image = self.imgTK
    #     self.label.grid( column = 0, row = 0 )
    #     self.window.geometry( str(self.originImg.width) + 'x' + str(self.originImg.height) )