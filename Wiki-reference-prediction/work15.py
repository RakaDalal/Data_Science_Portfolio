import sys
reload(sys) 
sys.setdefaultencoding('UTF8')
'''
#execfile('work14.py')
f=open("timeframe.txt")
f=f.read()
'''
citprof={}
for w in citnet.keys():
    print(citnet.keys().index(w))
    citprof[w]=[]
    for x in citnet[w]:
        if(x in biggy.keys()): 
            citprof[w].append(biggy[x])
        else:
            print("GONE")     

citp=open("timeframe2.txt",'w')
for w in citprof.keys():
    citp.write("A"+"   "+str(w)+'\n'*2)
    for x in citprof[w]:
           citp.write("B"+"   "+str(x)+'\n')
    citp.write('\n'*2)


