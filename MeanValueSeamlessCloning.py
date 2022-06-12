import numpy as np
from PIL import Image
import time


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
    print('computing mean value coordinate, boundaryVertex shape: , clock: ', np.array( sourceBoundaryVertex ).shape , time.clock() )

    # result = list()
    # for col in range( mask.shape[ 0 ] ):
    #     for row in range( mask.shape[ 1 ] ):
    #         if mask[ col , row ] == True:
    #             angles = CalAngle( col , row , sourceBoundaryVertex )             # shape: len( BoundaryVertex ) x 1
    #             weights = CalWeight( col , row , angles , sourceBoundaryVertex )  # shape: len( BoundaryVertex )
    #             # result[ 1 , col , row , : ] = weights
    #             result.append( (col,row,weights) )

    # 先取得non zero index
    nonZeroX, nonZeroY = np.nonzero(mask)

    # 邊界點數量
    boundaryVertexSize = np.array(sourceBoundaryVertex).shape[0]

    # 內部點數量
    innerPointSize = len(nonZeroX)

    # 把內部點座標改成[x ,y ,x ,y...]的形式
    arrayCoordinate = np.insert(nonZeroY, np.arange(innerPointSize), nonZeroX)

    # 先把sourceBoundaryVertex重複innerPointSize次(與內部點數量相同)
    sourceBoundaryVertex = np.array(sourceBoundaryVertex)               # 轉numpy
    flattenSourceBoundaryVertex = np.reshape(sourceBoundaryVertex, -1)  # 攤平。假設原始有4個boundary vertex, 就會變成4*2=8(x,y,x,y...)
    tileBoundary = np.tile(flattenSourceBoundaryVertex, innerPointSize) # 重複內部點數量次(在上述例子就是內部點數量 * 8)
    
    # 各自resahpe成(總內部點數量 * 對應值數量)的形狀
    totalBoundary = np.reshape(tileBoundary, (innerPointSize, -1))       # 座標點的攤平確保形狀正確(內部點數量 * 邊界點座標) => x, y
    totalCoordinate = np.reshape(arrayCoordinate, (innerPointSize, 2))   # 內部點座標(內部點數量 * 2) => x, y

    # 把兩個資料做Concat(先內部點座標，再邊界點座標)
    combineColumn = np.hstack((totalCoordinate, totalBoundary))

    # 算Angle & Weight
    # allWeight = np.apply_along_axis(CalAngle, 1, combineColumn)
    allWeight = CalAngle(combineColumn, boundaryVertexSize, innerPointSize)

    # 把內部點座標與結果Weight做Concat(內部點座標, Weight)
    combineCoordAngle = np.hstack((totalCoordinate, allWeight))

    # 轉成list內包含一堆tuple
    tupleCoorAngle = tuple(map(tuple, combineCoordAngle))
    listTuple = list(tupleCoorAngle)
    
    return listTuple

def CalAngle( combineColumn, boundaryVertexSize, innerPointSize ):

    # 直接把CalAngle整段做效能改善(apply_along_axis沒有用，超慢ㄇㄉ)
    dataPatchSize = combineColumn.shape[1]
    print('Check combineColumn shape', combineColumn.shape)
    print('Check innerPointSize', innerPointSize)
    print('Check dataPatchSize', dataPatchSize)

    # 把所有資料攤平
    flattenAll = np.reshape(combineColumn, -1)

    # 算phiArray(會有innerPointSize組資料)
    allCol = flattenAll[::dataPatchSize]    # pixel1 col, pixel2 col...
    allRow = flattenAll[1::dataPatchSize]   # pixel1 row, pixel2 row...

    # 取出每一組的boundary
    getBoundaryCondition = np.ones((flattenAll.shape), dtype=bool)
    getBoundaryCondition[::dataPatchSize] = False   # 不取出col位置資料
    getBoundaryCondition[1::dataPatchSize] = False  # 不取出row位置資料
    print("Check boundary condition", getBoundaryCondition[:20])

    boundaryVertex = flattenAll[getBoundaryCondition > 0]   # 只會有重複dataPatchSize的boundaryVertex(x, y, x, y...)
    print("Check boundaryVertex", boundaryVertex[:20])

    # 重複 allCol跟allRow boundaryVertexSize * 2次，再把兩者互斥的位置改成0
    # 假設4個邊界點，那麼就要[col1,    0,   col1,    0,    col1,    0, col1,    0]
    # 假設4個邊界點，那麼就要[0,    row1,   0,    row1,    0,    row1, 0,    row1]
    # boundaryVertex目前是:[ x,       y,   x,       y,    x,       y, x,       y]
    # 最後boundaryVertex與前面兩個相減得出正確的boundaryVertex用於算出phiArray
    repeatCol = np.repeat(allCol, boundaryVertexSize * 2)
    repeatRow = np.repeat(allRow, boundaryVertexSize * 2)

    forMinusCol = np.copy(repeatCol)
    forMinusRow = np.copy(repeatRow)

    forMinusCol[1::2] = 0
    forMinusRow[::2] = 0

    print('Check forMinusCol', forMinusCol[:20])
    print('Check forMinusRow', forMinusRow[:20])

    minusBoundaryVertex = boundaryVertex - forMinusCol - forMinusRow
    print('Check minusBoundaryVertex', minusBoundaryVertex[:20])

    # 算出正確的phiArray(長度是: 邊界點數量 * 總共內部pixel數量)
    minusBoundaryVertexX = minusBoundaryVertex[::2]
    minusBoundaryVertexY = minusBoundaryVertex[1::2]
    phiArray = np.arctan2(minusBoundaryVertexY, minusBoundaryVertexX)

    print('Check phiArray', phiArray[:20])

    # 算angles
    secondPhiArray = np.copy(phiArray)

    # 先取出第一個邊界點的值
    firstBoundary = np.copy(phiArray)
    firstBoundary = firstBoundary[::boundaryVertexSize]
    repeatFirstBoundary = np.repeat(firstBoundary, boundaryVertexSize)

    # 把除了最後一個batch元素之外的值都清0
    maskLastBoundaryIndex = np.zeros(phiArray.shape)
    maskLastBoundaryIndex[boundaryVertexSize - 1::boundaryVertexSize] = 1

    # 相乘後加回secondPhiArray
    maskRepeatFirst = repeatFirstBoundary * maskLastBoundaryIndex

    print('Check maskRepeatFirst', maskRepeatFirst[:20])

    # 陣列向左移動，然後把邊界點數量batch的最後一個變成0
    secondPhiArray = np.roll(secondPhiArray, -1)
    secondPhiArray[boundaryVertexSize - 1::boundaryVertexSize] = 0

    print('Before adding last boundary for last index(secondPhiArray)', secondPhiArray[:20])

    # 與maskRepeatFirst相加(設定最後一個欄位)
    secondPhiArray = secondPhiArray + maskRepeatFirst
    print('secondPhiArray result', secondPhiArray[:20])

    # 算angles
    angles = phiArray - secondPhiArray

    # 算weights
    print("Check minusBoundaryVertexX", minusBoundaryVertexX[:20])
    print("Check minusBoundaryVertexY", minusBoundaryVertexY[:20])
    norm = np.linalg.norm((minusBoundaryVertexX, minusBoundaryVertexY), axis=0)
    normArray = np.copy(norm)
    print('Check norm shape', normArray.shape)

    tanAngles = np.tan(angles / 2)
    print("Check tanAngles shape", tanAngles.shape)

    leftTanAngle = np.copy(tanAngles)

    # 陣列向右移動，然後把邊界點數量batch的第一個變成0
    rightTanAngle = np.roll(tanAngles, 1)
    rightTanAngle[::boundaryVertexSize] = 0

    print('Check leftTanAngle', leftTanAngle[:20])
    print('Check rightTanAngle after right roll', rightTanAngle[:20])

    # 把除了第一個batch元素之外的值都清0
    maskFirstBoundaryIndex = np.zeros(phiArray.shape)
    maskFirstBoundaryIndex[::boundaryVertexSize] = 1

    # 取出最後一個Batch元素並重複boundaryVertexSize次
    lastBoundaryAngle = np.copy(tanAngles)
    lastBoundaryAngle = lastBoundaryAngle[boundaryVertexSize - 1::boundaryVertexSize]
    repeatLastBoundaryAngle = np.repeat(lastBoundaryAngle, boundaryVertexSize)

    # 相乘後加回rightTanAngle
    maskRepeatLast = maskFirstBoundaryIndex * repeatLastBoundaryAngle

    rightTanAngle = rightTanAngle + maskRepeatLast

    print('Check correct rightTanAngle', rightTanAngle[:20])

    weights = (rightTanAngle + leftTanAngle) / normArray

    reshapeWeights = np.reshape(weights, (-1, boundaryVertexSize))
    weightSum = np.sum(reshapeWeights, axis=1)
    print('Check weightSum shape', weightSum.shape)

    repeatWeightSum = np.repeat(weightSum, boundaryVertexSize)
    repeatWeightSum = np.reshape(repeatWeightSum, (-1, boundaryVertexSize))
    print('Check repeatWeightSum shape', repeatWeightSum.shape)

    lambdaI = reshapeWeights / repeatWeightSum

    print('Check lambdaI shape', lambdaI.shape)

    return lambdaI

    '''
    # 以下是用apply_along_axis做的效能改善
    # 把col跟row分離出來
    col = dataPackage[0]
    row = dataPackage[1]

    # 把sourceBoundaryVertex分離出來
    sourceBoundaryVertex = dataPackage[2:]
    sourceBoundaryVertex = np.reshape(sourceBoundaryVertex, (-1, 2))

    Size = len( sourceBoundaryVertex )
    phiArray = []
    angles = []
  
    # for idx, ( x, y ) in enumerate ( sourceBoundaryVertex ):
    #     phiArray.append(np.arctan2(x-col, y-row))

    # 效能改善版本(phiArray)
    flattenSrouceBoundaryVertex = np.reshape(sourceBoundaryVertex, -1)
    xMinusCol = flattenSrouceBoundaryVertex[::2] - col
    yMinusRow = flattenSrouceBoundaryVertex[1::2] - row
    phiArray = np.arctan2(yMinusRow, xMinusCol)


    # for idx in range(Size):
    #     j = (idx+1) % Size
    #     angles.append(phiArray[idx] - phiArray[j])
    # angles = np.array(angles)

    # 效能改善版本(angles)
    secondPhiArray = phiArray[1:]
    secondPhiArray = np.append(secondPhiArray, phiArray[0])
    angles = phiArray - secondPhiArray


    # 算 weights
    weights = np.empty((len(sourceBoundaryVertex)))
    lambdaI = np.empty((len(sourceBoundaryVertex)))
    normArray = np.empty((len(sourceBoundaryVertex)))
    
    # for idx, ( x, y ) in enumerate ( sourceBoundaryVertex ):
    #     vec = np.array((x,y)) - np.array((col,row))
    #     norm = np.linalg.norm( vec )
    #     normArray[ idx ] = norm

    # 算weights的效能改善版本
    vec = np.insert(yMinusRow, np.arange(len(xMinusCol)), xMinusCol)
    vec = np.reshape(vec, (-1, 2))
    norm = np.linalg.norm(vec, axis=1)
    normArray = np.copy(norm)


    size = len(sourceBoundaryVertex)
    
    tanAngles = np.tan( angles / 2 )
    
    
    # for i in range( size ):
    #     leftIdx = ( i ) % size
    #     rightIdx = ( i - 1 ) % size
    #     weights[ i ] = ( tanAngles[ rightIdx ] + tanAngles[ leftIdx ] ) / normArray[ i ]

    # 效能改善版本
    leftTanAngle = np.copy(tanAngles)
    rightTanAngle = np.insert(tanAngles[:-1], 0, tanAngles[-1])
    weights = (rightTanAngle + leftTanAngle) / normArray


    weightSum = np.sum(weights)
    lambdaI = weights / weightSum
    
    return lambdaI

    '''

    '''
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

def CalWeight( coord , angles , sourceBoundaryVertex ):

    # 把col跟row分離出來
    col = coord[0]
    row = coord[1]

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

    return lambdaI

def CalDiff( source , target , sourceBoundaryVertex , centerCoord , offset ):
    print('cal diff along boundary')

    length = len( sourceBoundaryVertex )
    diffs = np.empty( ( length , 3 ), dtype=int )
    for idx , (col,row) in enumerate( sourceBoundaryVertex ):
        
        targetCol , targetRow = Source2TargetCoord( col , row , centerCoord , offset )
        
        diff = np.array(target[ targetCol , targetRow ], dtype=int) - np.array(source[ col , row ], dtype=int)
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
    sourceImg = np.array(sourceImg, dtype=int)
    targetImg = np.array(targetImg, dtype=int)
    length = len(targetBoundaryVertex)
    # for idx , (height,width,weights) in enumerate( lambdas ):
    for idx , data in enumerate( lambdas ):
        height = int(data[0])
        width = int(data[1])
        weights = []
        for i in range(length):
            weights.append(data[ i + 2 ])
        rX = 0
        targetHeight , targetWidth = Source2TargetCoord( height , width , centerCoord , offset )
        for boundaryIdx in range( len(targetBoundaryVertex) ):
            rX += diffs[ boundaryIdx ] * weights[ boundaryIdx ]
        intRx = np.array(rX, dtype=int)
        targetImg[ targetHeight , targetWidth , : ] = sourceImg[height, width, :] + (intRx)

    targetImg[ targetImg < 0 ] = 0
    targetImg[ targetImg > 255 ] = 255

    targetImg = np.array(targetImg, dtype=np.uint8)

    return targetImg