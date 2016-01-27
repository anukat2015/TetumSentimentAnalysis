'''
Created on Oct 7, 2014

@author: thiennguyen
'''

import glob, os
import numpy as np
import random
import pandas as pd
from pprint import pprint
#----------------------------------------------------------------------
class DocumentsAccess():          
    def readingDatabaseTetum(self, filePath, sheet, head = None):
        xl = pd.ExcelFile(filePath)
        df = xl.parse(sheet, header=head)
        df.head()
        return df
    
       
#----------------------------------------------------------------------
if __name__ == "__main__":

    filePath = "Database/Sentiment/Sentiment/PoliceRelations/positive.xlsx"
    sheet = "Sheet1"
    da = DocumentsAccess()
    pos = da.readingDatabaseTetum(filePath, sheet)
    pprint(pos[0][0])

    
