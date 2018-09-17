import sys
reload(sys) 
sys.setdefaultencoding('UTF8')
'''
nn=open('timesn.txt','w')
for w in range(len(time2)):
    nn.write("#index"+str(w)+"   "+r[w]+"\n")
    if(len(time2[w])>0):
          for times in range(len(time2[w])):
              nn.write("A"+"   "+str(time2[w][times])+"\n")
    nn.write("\n"*2)  
'''
nn=open('timesrefn.txt','w')
for w in range(len(time2)):
    nn.write("#index"+str(w)+"   "+r[w]+"\n")
    if(len(time2[w])>0):
          for times in range(len(time2[w])):
              nn.write("A"+"   "+str(refname2[w][times])+"     "+"B"+"       "+str(time2[w][times])+"\n")
    nn.write("\n"*2) 
