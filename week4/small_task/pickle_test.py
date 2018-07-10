import pandas
import pickle


path = r'C:\Users\bchaudhary\Desktop\16_Quarter_Output\1449_merchants.txt'


lis=[]

with open(path, 'r') as f:
    for line in f:
        line=line.strip()
        lis.append(line)
        


pickle_in = open("train_output.pkl","rb")
example_dict = pickle.load(pickle_in)
print (example_dict[:5])


def get_ygst_desc(row):

    digits = re.compile("\d")

    def tokenizer(desc):

        return digits.sub("0", "".join(desc.lower()))

 

    variant_str = tokenizer(row)

    variant_str = re.sub(' +',' ',variant_str)

    return variant_str


example_dict=example_dict.filter(items=['tag_sequence', 'description_unmasked','yodlee_name'])

example_dict=example_dict[example_dict['yodlee_name'].isin(lis)]

new_df=example_dict
#new_df.to_excel("after_given_funxtion.xlsx")
import re
new_df['ysgt'] =new_df['description_unmasked'].apply(lambda x:get_ygst_desc(x))

new_df=new_df.dropna()

def find_vendor(x):
    
     temp=x['tag_sequence'].split(" ")
     temp1= x['ysgt'].split(" ")
   
     s=""
     if(len(temp)==len(temp1)):
        
         
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
                 return s 
            
     else:
         return s
     
train_s=[]
#new_df[new_df['tag_sequence'].map(len) == new_df['ysgt'].map(len)]
print(new_df)
new_df['vendor_name']=new_df.apply(find_vendor,axis=1)

new_df=new_df[['yodlee_name','vendor_name']]

new_df=new_df.groupby(('yodlee_name'))['vendor_name'].apply(set)
new_df=new_df.reset_index()

filehandler = open("Final_training.obj","wb")
pickle.dump(new_df,filehandler)
filehandler.close()

new_df.to_excel("after_given_funxtion.xlsx")





