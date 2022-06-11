import enum
import math
from operator import truediv
from turtle import right
from cv2 import normalize
import numpy as np
from PIL import Image
from array import array
import cv2
import math


def RowCol2ColRow( sourceBoundaryVertex ):
    result = list()
    for i in range( len(sourceBoundaryVertex) ):
        x = sourceBoundaryVertex[ i ][ 0 ]
        y = sourceBoundaryVertex[ i ][ 1 ]
        result.append((y,x))
    return result

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
    sourceBoundary = RowCol2ColRow( sourceBoundaryVertex )
    centerCoordColRow = ( centerCoord[ 1 ] , centerCoord[ 0 ] )
    targetBoundary , offset = CalTargetBoundaryVertex( sourceBoundary , centerCoordColRow )

    lambdas = MeanValueCoordinate( sourcemask , sourceBoundary )                                # 每個內部點對於每個邊界點的權重 list: [ ( height, width, 邊界點數 ) ]
    diffs = CalDiff( sourceImg , targetImg , sourceBoundary , centerCoordColRow , offset )      # 每個邊界點的亮度差 shape: (邊界點數, 3)
    result = SeamlessCloning( sourceImg , targetImg , targetBoundary , lambdas , diffs , offset , centerCoordColRow )

    return Image.fromarray( result )
'''
return weights: ( width, height, 邊界點數 )
'''
def MeanValueCoordinate( mask , sourceBoundaryVertex ):
    print('computing mean value coordinate, boundaryVertex shape: ', np.array( sourceBoundaryVertex ).shape )

    result = list()
    for col in range( mask.shape[ 0 ] ):
        for row in range( mask.shape[ 1 ] ):
            if mask[ col , row ] == True:
                angles = CalAngle( col , row , sourceBoundaryVertex )             # shape: len( BoundaryVertex ) x 1
                weights = CalWeight( col , row , angles , sourceBoundaryVertex )  # shape: len( BoundaryVertex )
                # result[ 1 , col , row , : ] = weights
                result.append( (col,row,weights) )
    return result

def CalAngle( col , row , sourceBoundaryVertex ):
    matrixSize = len( sourceBoundaryVertex )
    dotMatA = np.zeros( ( matrixSize , matrixSize*2 ) )
    dotMatB = np.zeros( ( matrixSize*2 , matrixSize ) )
    dotMat = np.zeros( ( matrixSize , matrixSize ) )
    vecArray = []

    normMatA = np.zeros( ( matrixSize , matrixSize ) )
    normMatB = np.zeros( ( matrixSize , 1 ) )
    normMat = np.zeros( ( matrixSize , 1 ) )
    normArray = []
    
    for idx, ( x, y ) in enumerate ( sourceBoundaryVertex ):
        vec = np.array((x,y)) - np.array((col,row))
        norm = np.linalg.norm( vec )
        normArray.append( norm )
        vecArray.append( vec )

    for idx, ( x, y ) in enumerate ( sourceBoundaryVertex ):
        dotMatA[ idx , idx*2 ] = vecArray[ idx ][ 0 ] 
        dotMatA[ idx , idx*2 + 1 ] = vecArray[ idx ][ 1 ] 

        secondIdx = ( idx + 1 ) % matrixSize
        dotMatB[ idx*2 , idx ] = vecArray[ secondIdx ][ 0 ] 
        dotMatB[ idx*2 + 1 , idx ] = vecArray[ secondIdx ][ 1 ]

    for idx , norm in enumerate( normArray ):
        normMatA[ idx , idx ] = norm
        secondIdx = ( idx + 1 ) % matrixSize
        normMatB[ idx , 0 ] = normArray[ secondIdx ]
    # print("norm: ", norm )
    dotMat = np.matmul( dotMatA , dotMatB )
    normMat = np.matmul( normMatA , normMatB )
    normMat = 1 / normMat
    cosAngle = np.matmul( dotMat , normMat )
    angles = np.arccos( cosAngle )
    # angles = angles * 180 / np.pi
    # print("angles.shape", angles.shape)
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

def CalWeight( col , row , angles , sourceBoundaryVertex ):
    weights = np.empty((len(sourceBoundaryVertex)))
    lambdaI = np.empty((len(sourceBoundaryVertex)))

    normArray = np.empty((len(sourceBoundaryVertex)))
    
    for idx, ( x, y ) in enumerate ( sourceBoundaryVertex ):
        vec = np.array((x,y)) - np.array((col,row))
        norm = np.linalg.norm( vec )
        normArray[ idx ] = norm

    size = len(sourceBoundaryVertex)
    
    tanAngles = np.tan( angles / 2 )
    
    
    for i in range( size ):
        leftIdx = ( i ) % size
        rightIdx = ( i - 1 ) % size
        weights[ i ] = ( tanAngles[ rightIdx ] + tanAngles[ leftIdx ] ) / normArray[ i ]

    weightSum = np.sum(weights)
    lambdaI = weights / weightSum
    # print('angles: ',angles)
    # print('tanangles',tanAngles)
    # print('weights: ', lambdaI )
    return lambdaI

def CalDiff( source , target , sourceBoundaryVertex , centerCoord , offset ):
    print('cal diff along boundary')

    length = len( sourceBoundaryVertex )
    diffs = np.empty( ( length , 3 ) )
    for idx , (col,row) in enumerate( sourceBoundaryVertex ):
        
        targetCol , targetRow = Source2TargetCoord( col , row , centerCoord , offset )
        
        diff = target[ targetCol , targetRow ] - source[ col , row ]
        diffs[ idx ] = diff

    return diffs

def CalTargetBoundaryVertex( sourceBoundaryVertex , centerCoord ):
    length = len( sourceBoundaryVertex )
    offsetRow = 0
    offsetCol = 0
    for i in range( length ):
        offsetCol += sourceBoundaryVertex[ i ][ 0 ] / length
        offsetRow += sourceBoundaryVertex[ i ][ 1 ] / length
    
    targetBoundaryVertex = []
    for i in range( length ):
        sourceCol , sourceRow = sourceBoundaryVertex[ i ]
        targetCol = sourceCol - int( offsetCol ) + centerCoord[ 0 ]
        targetRow = sourceRow - int( offsetRow ) + centerCoord[ 1 ]
        targetBoundaryVertex.append( ( targetCol , targetRow ) )

    return targetBoundaryVertex , ( offsetCol , offsetRow )

def Source2TargetCoord( col , row , centerCoord , offset ):
    resultCol = col - offset[ 0 ] + centerCoord[ 0 ]
    resultRow = row - offset[ 1 ] + centerCoord[ 1 ]
    return ( int(resultCol) , int(resultRow) )

def SeamlessCloning( sourceImg , targetImg , targetBoundaryVertex , lambdas , diffs , offset , centerCoord ):
    print('start seamless Cloning')
    for idx , (height,width,weights) in enumerate( lambdas ):
        rX = 0
        targetHeight , targetWidth = Source2TargetCoord( height , width , centerCoord , offset )
        for boundaryIdx in range( len(targetBoundaryVertex) ):
            rX += diffs[ boundaryIdx ] * weights[ boundaryIdx ]
        targetImg[ targetHeight , targetWidth , : ] = sourceImg[ height , width , : ] + rX

    return targetImg