import re
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
f=open("tired.txt")
f=f.read()
lines=f.split('\n')
rec={}
for line in lines:
    h=line.split('   ')
    if(len(h)>1):
            if h[0] not in rec.keys():
                  rec[h[0]]=[]
                  rec[h[0]].append(h[1])
            else:
                  rec[h[0]].append(h[1])
   
g=open("timeframe2.txt")
g=g.read()
records=g.split('A')
rec2={}
for record in records:
    lines=record.split('\n')
    for i  in range(len(lines)):
       lines[i]=re.sub("[^0-9]", "",lines[i])
       lines[i]="".join(lines[i].split()) 
    rec2[lines[0]]=[]
    for x in range(1,len(lines)):
       if(len(lines[x])==4):
            rec2[lines[0]].append((int)(lines[x]))

for x in range(len(rec.keys())):
    if rec.keys()[x] in rec2.keys():
       plott=Counter(rec2[rec.keys()[x]])
       #Counter={int(k):int(v) for k in Counter.keys()}
       #plott.keys()=sorted(plott.keys())
       ea=plott.keys()
       ea=sorted(ea)
       plott2={}
       for xx in ea:
           plott2[xx]=plott[xx]
           
       plt.figure(x)
       X = np.arange(len(plott2.keys()))*15
       plt.bar(X,plott2.values(), align='center',width=3)
       plt.xticks(X, plott2.keys(),rotation=90)
       ymax = max(plott.values()) + 1
       plt.annotate(str(rec[rec.keys()[x]]), xy=(0,ymax*0.8),  xycoords='data')
       plt.ylim(0, ymax)
       plt.xlabel('Year')
       plt.ylabel('Citations')
       plt.title('Citation Profile')
       #plt.axis([0, 100, 0, 30])
       plt.grid(True)
       #plt.show()
       #fig=ax.get_figure()
       filename="file"+str(x)+".png"
       plt.savefig(filename)
    



        
        
     

