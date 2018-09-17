
execfile('work13.py')

import re
citnet={}
g=open("in.txt")
g=g.read()
g=g.split('\n')
for ind in g:
    if(ind.startswith('#index')):
         g[g.index(ind)]=ind[6:]
    else:
         g.remove(ind)

print len(g) 
print g[0]    
for i in range(7):
    print(i)
    f=open("citesa"+chr(97+i))
    f=f.read()
    network=f.split('\n')
    for p in network:
        ids=p.split()
        if(len(ids)==2):
            if(ids[1] in g):
                if(ids[1] in citnet.keys()):
                      citnet[ids[1]].append(ids[0])
                else:
                      citnet[ids[1]]=[]
                      citnet[ids[1]].append(ids[0])
                print(len(citnet.keys()))


     
