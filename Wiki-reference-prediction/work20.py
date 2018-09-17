import matplotlib.pyplot as plt
import numpy as np

linkr=open("link.txt")
linkr=linkr.read()
linkr=linkr.split('\n')
#linkr=linkr[1:]

#ff=open("/home/raka/stats/exp2012"+".txt",'w')
gg=open("/home/raka/stats/exp2012"+".txt",'w')
w20ro=open("/home/raka/stats/ref2012"+".txt",'w')
w20rn=open("/home/raka/stats/ref2014"+".txt",'w')
n=0

newr=[]
oldr=[]


for filename in linkr:
        w20v=linkr.index(filename) 
        newr.append([])
        oldr.append([])
        if(filename==''):
            execfile('work22.py')
            break 
	url=filename
	threshold=5
	stat1refinc=[]
	stat2comauth=[]
	stat3comcontx=[]
	stat4uncomcontx=[]
        stat5year={} 
        filename=filename.split(':')[2]
        w20ro.write("\n"*3+"GENRE:-  "+filename+"\n"*3)
        w20rn.write("\n"*3+"GENRE:-  "+filename+"\n"*3)
        gg.write("\n"*3+"GENRE:-  "+filename+"\n"*3)     
	execfile('work18.py')
	plt.hold(False)
	z=plt.hist(stat1refinc,bins=5)
	plt.xlabel("percentage increase in references")
	plt.ylabel("number of pages")
	plt.ylim(0,max(z[0])+2)
	plt.savefig("/home/raka/stats/"+filename+"stat1.png")
	plt.hold(False)
	z=plt.hist(stat2comauth,bins=5)
	plt.xlabel("percentage of common authors")
	plt.ylabel("number of pages")
	plt.ylim(0,max(z[0])+2)
	plt.savefig("/home/raka/stats/"+filename+"stat2.png")
	plt.hold(False)
	z=plt.hist(stat3comcontx,bins=5)
	plt.xlabel("percentage of common contexts")
	plt.ylabel("number of pages")
	plt.ylim(0,max(z[0])+2)
	plt.savefig("/home/raka/stats/"+filename+"stat3.png")
	plt.hold(False)
	z=plt.hist(stat4uncomcontx,bins=5)
	plt.xlabel("percentage of uncommon contexts")
	plt.ylabel("number of pages")
	plt.ylim(0,max(z[0])+2)
	plt.savefig("/home/raka/stats/"+filename+"stat4.png")
	ea=stat5year.keys()
	ea=sorted(ea)
        values=[]
        for xx in ea:
            values.append(stat5year[xx])
	#plt.figure(x)
	plt.hold(False)
        X = np.arange(len(ea))*15
	plt.bar(X,values, align='center',width=3)
	plt.xticks(X,ea,rotation=90)
	ymax = max(values) + 1
	plt.ylim(0, ymax)
	plt.xlabel('Year')
	plt.ylabel('No of pages')
	#plt.axis([0, 100, 0, 30])
	plt.grid(True)
	#plt.show()
	#fig=ax.get_figure()
	plt.savefig("/home/raka/stats/"+filename+"stat5.png")
