# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:06:51 2018

@author: bchaudhary
"""

import pandas as pd

import json
import glob

  
path = r'C:\Users\bchaudhary\Desktop\16_Quarter_Output\1449_merchants.txt'


lis=[]

with open(path, 'r') as f:
    for line in f:
        line=line.strip()
        lis.append(line)
        



import zipfile 
 
l=['NETFLIX','APPARELSAVE']
train = pd.DataFrame()
count=0
with zipfile.ZipFile("16_quarter_1400_merch_with_YGST.zip", "r") as z:
  for filename in z.namelist():
     if(filename == "16_quarter_1400_merch_with_YGST/16_quarter_1437_20.output.json"):
        print(filename)  
        with z.open(filename) as f:  
         data = f.read()  
         d = json.loads(data.decode("utf-8"))

         d = pd.DataFrame(d['transaction_list'],dtype=object)
#             train = (pd.DataFrame.from_dict(word, orient='index'))
         train = train.append(d,ignore_index=True)
         train=train[train['tde_yodlee_name'].isin(lis)]
         
    
         import re
         def get_ygst_desc(row):
        
             digits = re.compile("\d")
        
             def tokenizer(desc):
        
                 return digits.sub("0", "".join(desc.lower()))
        
         
        
             variant_str = tokenizer(row)
        
             variant_str = re.sub(' +',' ',variant_str)
        
             return variant_str
        
        
         train['ysgt'] =train['description'].apply(lambda x:get_ygst_desc(x))              
           		
        
         train=train.dropna()
        
         def find_vendor(x):
            
             temp=x['tag_sequence'].split(" ")
             temp1= x['ysgt'].split(" ")
        #     print (temp)
        #     print(temp1[1])
             ans=[]
             if(len(temp)==len(temp1)):
                
                 s=""
                 count=0
                 for i in range(len(temp)):
        
                     if(temp[i]=='<vendor_name>' and count==0):
                         s=temp1[i]+" "
                         count=1
                     elif(temp[i]=='<vendor_name>' and count==1):
                         s=s+temp1[i]+" "
                         count=1
                     elif(count==1):
                         s=s[:-1]
                         ans.append(s)
                         s=""
                         count=0
                 return ans  
             else:
                 return ans
        

         train['vendor_name']=train.apply(find_vendor,axis=1)
         train=train[['tde_yodlee_name','vendor_name','16_Quarter_Count'] ]           
         
         train.to_excel("check.xlsx")
         
         train=train.groupby(('tde_yodlee_name')).apply(sum)  

         train=train.reset_index()


final_ans=final_ans.groupby(('tde_yodlee_name'))['vendor_name','16_Quarter_Count'].apply(sum)  

final_ans=final_ans.reset_index()

final_ans=final_ans.rename(columns={'tde_yodlee_name':'yodlee_name','vendor_name':'vendor_name1'})



import pickle

pickle_in = open("Fruits.obj","rb")
data1 = pickle.load(pickle_in)

print(data1[:3])
#print(type(data1['vendor_name'][0]))

let= pd.merge(data1,final_ans,on='yodlee_name')

#print (let['vendor_name1'][0])
#
#string_list=let['vendor_name'][0]
#print (string_list[1])
#string2=let['vendor_name1'][0]
#string_list=string_list +string2

def ans(x):
    l3=x['vendor_name']
    l6=x['vendor_name1']
    
    def string_set(s1,string_list):
        return set(i for i in string_list 
                   if not any(i in s for s in s1))
    l4=string_set(l3,l6)
    l5=set(x['vendor_name'])
    return l4-l5    

let=let.sort_values('16_Quarter_Count')
let['difference']=let.apply(ans,axis=1)

let.to_excel("small_problem_ans.xlsx")


          
         