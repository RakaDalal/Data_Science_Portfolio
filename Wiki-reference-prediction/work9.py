l=[]
asb=[]
s=[]
ss=[]
for i in range(32):
        print(i)
        if(i<9):
	   root="disk"+str(i+1)+".gsd00"+str(i+1)
	   f=open(root)
	   ss=f.read()
	   asb=ss.split('#index')
	   for w in asb:
		ww=w.split('\n')
		for x in ww:
			if x.startswith('#f'):
				l.append(x)
        else:
           root="disk"+str(i+1)+".gsd0"+str(i+1)
	   f=open(root)
	   ss=f.read()
	   asb=ss.split('#index')
	   for w in asb:
		ww=w.split('\n')
		for x in ww:
			if x.startswith('#f'):
				l.append(x) 

l=list(set(l))

'''
g=open("F:\\security3.txt",'w')
for i in range(2):
	z="#fsecurity_and_privacy"
	asb=[]
	s=[]
	ss=[]
	root="F:\\disk"+str(i+2)+".gsd00"+str(i+2)
	f=open(root)
	ss=f.read()
	asb=ss.split('#index')
	for w in asb:
		ww=w.split('\n')
		l=[x for x in ww if x.startswith('#f')]
		if l:
		     if(l[0]==z):
		               g.write('\n'+w+'\n')
'''
		               
