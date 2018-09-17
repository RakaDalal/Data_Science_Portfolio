f=open("map_inf4.txt")
a=f.read()
sites=a.split('\n'*5)
r=list()
x=[]
for w in sites:
   refs=w.split('REFERENCE')
   for g in refs:
        r.append([])
        lines=g.split('\n')
        for line in lines:
             if(line!=None):
                 if(line.startswith('^ ')):
                     r[refs.index(g)].append(line)
 		 if(line.startswith('#y')):	
                     print(2013-int(line[2:-1]))
                     r[refs.index(g)].append(2013-int(line[2:-1]))
                     if(r[refs.index(g)][len(r[refs.index(g)])-3]=='X'):   
                            x.append(float(float(r[refs.index(g)][len(r[refs.index(g)])-2])/float(r[refs.index(g)][len(r[refs.index(g)])-1])))
                 if(line.startswith('#n')):
                     r[refs.index(g)].append('X')   
                     r[refs.index(g)].append(line[2:-1]) 


import matplotlib.pyplot as plt
n,bins,patches = plt.hist(x,5,facecolor='g',alpha=0.75)
plt.xlabel('Average citations')
plt.ylabel('No of Papers cited')
plt.title('Average citations')
plt.axis([0, 25, 0, 15])
plt.grid(True)
plt.show()
