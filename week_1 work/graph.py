# -*- coding: utf-8 -*-
"""
Created on Tue May 22 19:48:50 2018

@author: bchaudhary
"""

import numpy as np

import pandas as pd
import zipfile


path = r'C:\Users\bchaudhary\Desktop\16_Quarter_Output\1449_merchants.txt'


lis=[]

with open(path, 'r') as f:
    for line in f:
        line=line.strip()
        lis.append(line)
        
    
print(lis)
   
import glob   
path = r'C:\Users\bchaudhary\Desktop\16_Quarter_Output\BANK\*.gz'   

path1 = r"C:\Users\bchaudhary\Desktop\16_Quarter_Output\CARD\*"

r=0

files=glob.glob(path)   
for file in files:     
    print(file)
    start=file.find("BANK")+5
    #print(file[start:start+6])
    d=file[start:start+8]
    df=pd.read_csv(file, sep='|',usecols=[34], error_bad_lines=False,names=['YODLEE_MERCHANT_NAME'])
    #df = df.rename(columns={'7-ELEVEN':'YODLEE_MERCHANT_NAME'})
    #print(df)
    for i in df.iterrows():
        #print(i[1][0])
        if(i[1][0]) not in lis:
            df=df[df.YODLEE_MERCHANT_NAME!=i[1][0]]
    path2=path1+file[start:start+8]  
    #print(path2)
    path2 = path2+'*.gz'   
    #print(path2)
    files1=glob.glob(path2)   
    for f in files1:     
        print(f)
        if(f):
            gf=pd.read_csv(f, sep='|',usecols=[34], error_bad_lines=False,names=['YODLEE_MERCHANT_NAME'])
            
            for i in gf.iterrows():
                if(i[1][0]) not in lis:
                    gf=gf[gf.YODLEE_MERCHANT_NAME!=i[1][0]]
                    
            frames =[df,gf]
            df_=pd.concat(frames)
    df_final= df_.groupby(['YODLEE_MERCHANT_NAME']).size().reset_index(name='counts')
    
    date=np.full(len(df_final.index),d)
    #print(date)
    
    df_final['Date']=  date   
    if(r==0):
        df_last=df_final
        r+=1
    else:    
        frames1=[df_last,df_final]
        df_last=pd.concat(frames1)
    
    
    
    
#print(df_last)


import matplotlib.pyplot as plt

df_new = df_last.sort_values(['YODLEE_MERCHANT_NAME','Date'],ascending=[True, True])

df_new.to_excel("data1.xlsx")

print(df_new)    

m=0
k=0
for row in df_new.iterrows():
    if(k!=row[1]['YODLEE_MERCHANT_NAME']):
        if(m!=0):
            fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
            ax.plot(x, y)
            plt.title('Analysis of the merchant :'+k)
            plt.ylabel("Date")
            plt.xlabel("counts")
            plt.show()
           
            fig.savefig(k+'.png')   # save the figure to file
            plt.close(fig)
        k=row[1]['YODLEE_MERCHANT_NAME']
        x=[]
        y=[]
        x.append(row[1]['Date'])
        y.append(row[1]['counts'])
        m+=1
    else:
        x.append(row[1]['Date'])
        y.append(row[1]['counts'])
    
    



