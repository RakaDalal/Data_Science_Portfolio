biggy=[]
for i in range(33):
        print(i)
        if(i<9):
	   root="disk"+str(i+1)+".gsd00"+str(i+1)
	   f=open(root)
	   ss=f.read()
	   asb=ss.split('\n'*2)
	   for w in asb:
                flag=0
		ww=w.split('\n')
                h=[]
		for x in ww:
			if x.startswith('#*'):
				h.append(x[2:])
                        if x.startswith('#index'):
                                h.append(x)
                biggy.append(h)  
        else:
           root="disk"+str(i+1)+".gsd0"+str(i+1)
	   f=open(root)
	   ss=f.read()
	   asb=ss.split('\n'*2)
	   for w in asb:
                flag=0
		ww=w.split('\n')
                h=[]
		for x in ww:
			if x.startswith('#*'):
				h.append(x[2:])
                        if x.startswith('#index'):
                                h.append(x)
                biggy.append(h)  

