import os

file_name=os.listdir('cg')
for i in file_name:
    with open('cg\\'+i,'r') as f:
        data=f.readlines()
        for j,line in enumerate(data):
            with open('edge_list_'+1+'.txt','a') as f1:
                line=str(line).replace('\n',' ')
                f1.write(line)
