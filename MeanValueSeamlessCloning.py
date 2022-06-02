import enum
import math
from operator import truediv
from turtle import right
import numpy as np
from PIL import Image
from array import array
import cv2
import math


class BoundaryInfo:
    def __init__(self, coord , length , leftAngle , rightAngle ) :
        self.coord = coord # np array(x,y)
        self.length = length
        self.leftAngle = leftAngle
        self.rightAngle = rightAngle

class PixelInfo:
    def __init__( self, ) :
        print('test')

'''
source: pil
sourcemask: numpy
sourceBoundaryVertex: list
target: pil
centerCoord: tuple
'''
def Start( source ,  sourcemask , sourceBoundaryVertex , target , centerCoord ):
    # 轉成 Numpy
    sourceImg = np.array( source )
    targetImg = np.array( target )
    MeanValueCoordinate( sourcemask , sourceBoundaryVertex )

def MeanValueCoordinate( mask , BoundaryVertex ):
    print('computing mean value coordinate')
    size = np.where( mask == True ).count
    resultAngles = np.empty( ( mask.shape[ 0 ] , mask.shape[ 1 ] , len( BoundaryVertex ) ) )
    resultweights = np.empty( ( mask.shape[ 0 ] , mask.shape[ 1 ] , len( BoundaryVertex ) ) )
    for col in range( mask.shape[ 0 ] ):
        for row in range( mask.shape[ 1 ] ):
            if mask[ col , row ] == True:
                angles = CalAngle( col , row , BoundaryVertex )             # len( BoundaryVertex ) x 1
                # weights = CalWeight( col , row , angles , BoundaryVertex )  # len( BoundaryVertex ) x 1
                resultAngles[ col , row , : ] = angles[:,0]
                # resultweights[ col , row , : ] = weights[:,0]

def CalAngle( col , row , BoundaryVertex ):
    matrixSize = len( BoundaryVertex )
    dotMatA = np.zeros( ( matrixSize , matrixSize*2 ) )
    dotMatB = np.zeros( ( matrixSize*2 , matrixSize ) )
    dotMat = np.zeros( ( matrixSize , matrixSize ) )
    vecArray = []

    normMatA = np.zeros( ( matrixSize , matrixSize ) )
    normMatB = np.zeros( ( matrixSize , 1 ) )
    normMat = np.zeros( ( matrixSize , 1 ) )
    normArray = []
    
    for idx, ( x, y ) in enumerate ( BoundaryVertex ):
        vec = np.array((x,y)) - np.array((col,row))
        # print('new vec: ',vec)
        norm = np.linalg.norm( vec )
        normArray.append( norm )
        vecArray.append( vec )

    for idx, ( x, y ) in enumerate ( BoundaryVertex ):
        dotMatA[ idx , idx*2 ] = vecArray[ idx ][ 0 ] 
        dotMatA[ idx , idx*2 + 1 ] = vecArray[ idx ][ 1 ] 

        secondIdx = ( idx + 1 ) % matrixSize
        dotMatB[ idx*2 , idx ] = vecArray[ secondIdx ][ 0 ] 
        dotMatB[ idx*2 + 1 , idx ] = vecArray[ secondIdx ][ 1 ]

    for idx , norm in enumerate( normArray ):
        normMatA[ idx , idx ] = norm
        secondIdx = ( idx + 1 ) % matrixSize
        normMatB[ idx , 0 ] = normArray[ secondIdx ]

    dotMat = np.matmul( dotMatA , dotMatB )
    normMat = np.matmul( normMatA , normMatB )
    normMat = 1 / normMat
    cosAngle = np.matmul( dotMat , normMat )
    angles = np.arccos( cosAngle )
    angles = angles * 180 / np.pi
    print("angles.shape", angles.shape)
    return angles

    '''
    # for i in range( len( BoundaryVertex ) ):
    #     boundaryCoord = np.array( BoundaryVertex[ i ] )
    #     innerCoord = np.array( ( col , row ) )

    #     # test1 = np.array( BoundaryVertex[ 1 ] )
    #     # test2 = np.array( BoundaryVertex[ 2 ] )
    #     # print('origin: ', boundaryCoord - innerCoord )
    #     # print('angle1: ', CalAngle( boundaryCoord - innerCoord , test1 - innerCoord ) )
    #     # print('angle2: ', CalAngle( test1 - innerCoord , test2 - innerCoord ) )

        
    #     if i+1 > len( BoundaryVertex ) - 1:
    #         LboundaryCoord = np.array( BoundaryVertex[ 0 ] )
    #     else:
    #         LboundaryCoord = np.array( BoundaryVertex[ i + 1 ] )
    #     if i-1 < 0:
    #         RboundaryCoord = np.array( BoundaryVertex[ len( BoundaryVertex ) - 1 ] )
    #     else:
    #         RboundaryCoord = np.array( BoundaryVertex[ i - 1 ] )

        
    #     centerVec = boundaryCoord - innerCoord
    #     dist = np.sqrt( centerVec.dot( centerVec ) )
    #     leftVec = LboundaryCoord - innerCoord
    #     rightVec = RboundaryCoord - innerCoord
    #     # print('boundaryCoord: {} , innerCoord: {} , centerVec: {} '.format(boundaryCoord,innerCoord,centerVec))
    #     # print('boundaryCoord: {} , innerCoord: {} , rightVec: {} '.format(boundaryCoord,innerCoord,rightVec))
    #     # print('boundaryCoord: {} , innerCoord: {} , leftVec: {} '.format(boundaryCoord,innerCoord,leftVec))
    #     # if int(np.linalg.norm( centerVec )) == 0 or int(np.linalg.norm( leftVec )) == 0 or int(np.linalg.norm( rightVec )) == 0:
    #     #     continue
    #     leftAngle = CalAngle( centerVec , leftVec )
    #     rightAngle = CalAngle( rightVec , centerVec )
    #     # if int(leftAngle) == 180 or int(rightAngle) == 180:
    #     #     continue
    #     # print('angle old: ', leftAngle,rightAngle)
    #     boundaryInfo = BoundaryInfo( boundaryCoord , dist , leftAngle , rightAngle )

def CalAngle( vec1 , vec2 ):
    Lx = np.linalg.norm( vec1 )
    Ly = np.linalg.norm( vec2 )

    cos_angle = vec1.dot( vec2 )/( Lx * Ly )

    angle = np.arccos( cos_angle )
    result = angle * 180 / np.pi
    return result

    '''

def CalWeight( col , row , angles , BoundaryVertex ):
    print('cal weight')
    weights = np.empty((len(BoundaryVertex)))
    lambdaI = np.empty((len(BoundaryVertex)))

    size = len(BoundaryVertex)
    # 有問題
    length = np.linalg.norm( np.array( BoundaryVertex ) - np.array( ( col , row ) ) )
    tanAngles = np.tan( angles / 2 )
    
    for i in range( size ):
        leftIdx = ( i + 1 ) % size
        rightIdx = ( i - 1 ) % size
        weights[ i ] = ( tanAngles[ leftIdx ] + tanAngles[ rightIdx ] ) / length

    lambdaI = weights / np.sum( weights )

    return lambdaI

# def CalDiff( col , row , BoundaryVertex ):

