import torch
import numpy as np
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CalWeight(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        print('x shape: ', x.size() )
        allangle = []
        for i in range(x.size( dim = 0) ):
            col = x[i,0]
            row = x[i,1]
            sourceBoundaryVertex = x[i,2:]
            sourceBoundaryVertex = torch.reshape(sourceBoundaryVertex, (-1, 2))

            # Size = sourceBoundaryVertex.size( dim = 0 )

            flattenSrouceBoundaryVertex = torch.flatten(sourceBoundaryVertex)
            xMinusCol = flattenSrouceBoundaryVertex[::2] - col
            yMinusRow = flattenSrouceBoundaryVertex[1::2] - row
            phiArray = torch.atan2(yMinusRow, xMinusCol)            # ?

            secondPhiArray = phiArray[1:]
            phi = torch.tensor([phiArray[0]])
            print('secondPhiArray, phiArray: ' , secondPhiArray , phi )
            secondPhiArray = torch.cat( ( secondPhiArray, phi ) )
            print('secondPhiArray ' , secondPhiArray )

            angles = phiArray - secondPhiArray
            allangle.append(angles)

        return  allangle