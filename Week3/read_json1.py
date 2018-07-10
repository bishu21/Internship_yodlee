import pandas as pd

import json
import glob


#print (pd.read_json('/home/bishu/Desktop/small_task/16_quarter_1400_merch_with_YGST/16_quarter_1437_0.output.json'))


# l1 = glob.glob("/home/bishu/Desktop/small_task/16_quarter_1400_merch_with_YGST/*.json")
# print (l1)
  
import zipfile  
l=['ADVENTURE ISLAND']

count=0
with zipfile.ZipFile("16_quarter_1400_merch_with_YGST.zip", "r") as z:
   for filename in z.namelist():
   		
   		if(filename not in "16_quarter_1400_merch_with_YGST/"):
   			print(filename)  
   			gf=pd.read_json(filename)
   			gf=gf[['description_unmasked','yodlee_name','tag_sequence']]
   			gf[gf['yodlee_name'] is in (l)]
   			gf=gf.head(3)

   			if(count==0):
   				final=gf
   			else:	
   				frame=[final,gf]
   				final=pd.concat(frame)
   			print(len(final))
   			count+=1
   			if(count==1):
   				break

final['ysgt']=final['description_unmasked'].apply(lambda x:get_ygst_desc(x)) 

f=final.dropna()

def find_vendor(x):
    
    temp=x['tag_sequence'].split(" ")
    temp1= x['ysgt'].split(" ")
#     print (temp)
#     print(temp1[1])

    if(len(temp)==len(temp1)):
        ans=[]
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

train_s=[]
#f[f['tag_sequence'].map(len) == f['ysgt'].map(len)]
train_s.append(final[['tag_sequence','ysgt']].apply(find_vendor,axis=1))

import itertools
train_s = list(itertools.chain(*train_s))
merged = list(itertools.chain(*train_s))
final_train1=set(merged)

final_train1-final

for word in final_train1:
	if(word not in final)


print ("dfs")





#final=pd.concat(gf,ignore_index=True)