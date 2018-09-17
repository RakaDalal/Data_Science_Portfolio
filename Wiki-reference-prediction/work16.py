f2=open("timesrefn.txt")
f2=f2.read()
refwork=f2.split('#index')
refname2=[]
time2=[]
for p in refwork:
    refname2.append([])
    time2.append([])
    refs=p.split('A   ')
    for h in refs:
        if(h.startswith('^')):
            h1=h.split('B       ')[0]
            h2=h.split('B       ')[1]
            refname2[refwork.index(p)].append(h1)
            time2[refwork.index(p)].append(h2)
for i in range(len(refname2)):
    print(len(refname2[i])) 
