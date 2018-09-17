import re
stat=open("/home/raka/stats/Statistics_"+filename+".txt",'a')


def common(p,q):
     p=str(re.sub(r'[^\w]',' ',p))
     p=p.lower()
     q=str(re.sub(r'[^\w]',' ',q))
     q=q.lower()
     temp=len(list(set(p.split()).intersection(set(q.split()))))
     if(min(len(p.split()),len(q.split()))*1.0>0): 
       return temp/(min(len(p.split()),len(q.split()))*1.0)
     else:
       return 0.0

for r1 in refs2014:
    for r2 in refs2012:
       if(common(r1,r2)>0.8):
           refs2014.remove(r1)
           break  

newr[w20v][w18v].append(refs2014)
oldr[w20v][w18v].append(refs2012)

w20ro.write("***"+a2+'\n'*4)
w20rn.write("***"+a2+'\n'*4)
stat.write(a2+'\n'*4)
stat.write("No of new references 2014:" +str(len(refs2014))+"   "+ "2012:" +str(len(refs2012))+'\n'*2)
if(counter2>0):
    stat1refinc.append((float)((counter1)*100)/(counter2))
stat.write("Authors in 2014 references:-"+"\n")
names2014=[]
names2012=[]



for r1 in newr[w20v][w18v][0]:
    w20rn.write("A"+r1+'\n')
w20rn.write('\n'*3)

for r1 in oldr[w20v][w18v][0]:
    w20ro.write("A"+r1+'\n')
w20ro.write('\n'*3)

for r1 in refs2014:
    if(r1.startswith("^ a b ")):
       r1=r1[6:]
       if("(" in r1):
           r2=r1.split("(")[0]
           r3=r2.split(";")
           for name in r3:
               name=name.split(",")
               if(len(name)>1):
                    if(len(name[0])>0 and len(name[1])>0): 
                         stat.write(str(name[1])+" "+str(name[0])+"\n")
                         names2014.append(str(name[1])+" "+str(name[0])) 
    elif(r1.startswith("^ ")):
       r1=r1[2:]
       if("(" in r1):
           r2=r1.split("(")[0]
           r3=r2.split(";")
           for name in r3:
               name=name.split(",")
               if(len(name)>1):
                    if(len(name[0])>0 and len(name[1])>0): 
                         stat.write(str(name[1])+" "+str(name[0])+"\n")
                         names2014.append(str(name[1])+" "+str(name[0]))




stat.write("Authors in 2012 references:-"+"\n")
for r1 in refs2012:
    if(r1.startswith("^ a b ")):
       r1=r1[6:]
       if("(" in r1):
           r2=r1.split("(")[0]
           r3=r2.split(";")
           for name in r3:
               name=name.split(",")
               if(len(name)>1):
                    if(len(name[0])>0 and len(name[1])>0): 
                         stat.write(str(name[1])+" "+str(name[0])+"\n")
                         names2012.append(str(name[1])+" "+str(name[0]))
    elif(r1.startswith("^ ")):
       r1=r1[2:]
       if("(" in r1):
           r2=r1.split("(")[0]
           r3=r2.split(";")
           for name in r3:
               name=name.split(",")
               if(len(name)>1):
                    if(len(name[0])>0 and len(name[1])>0): 
                         stat.write(str(name[1])+" "+str(name[0])+"\n")
                         names2012.append(str(name[1])+" "+str(name[0]))      




gn=soup2.find_all('li')
context2012=[]
bn=[]
for wn in gn:
	bn.append(wn.get('id'))
g2n=soup2.find_all('p')
for eln in g2n:
	kn=[]
	for wn in eln.find_all('a'):
		if(wn!=None):
		     if((wn.get('href')).startswith('#cite')):
			   kn.append(wn.get('href'))
	for wwn in kn:
		for ssn in bn:
	               if(ssn!=None):
			      if(ssn in wwn):
                                    context2012.append(str(eln.text))

context2012=list(set(context2012))
gn=soup3.find_all('li')
common_context=0
context2014=[]
bn=[]
ren=[]
for wn in gn:
	bn.append(wn.get('id'))
        ren.append(wn)
g2n=soup3.find_all('p')
for eln in g2n:
	kn=[]
	for wn in eln.find_all('a'):
		if(wn!=None):
		     if((wn.get('href')).startswith('#cite')):
			   kn.append(wn.get('href'))
	for wwn in kn:
		for ssn in bn:
	               if(ssn!=None):
			      if(ssn in wwn and ren[bn.index(ssn)].text in refs2014):
                                    context2014.append(str(eln.text)) 
                                    for element in context2012:
                                         if(common(element,str(eln.text))>0.6 and len(element.split())>8 and len(str(eln.text).split())>8):
                                              print(eln.text+"\n"*2+element+"\n"*2) 
                                              common_context=common_context+1
                                              break 

if(len(names2014)>0):
       stat2comauth.append((float)(len(set(names2014).intersection(set(names2012)))*100)/len(set(names2014)))
if(len(context2014)>0):
       stat3comcontx.append((float)(common_context*100)/(len(context2014)))
       stat4uncomcontx.append((float)(100*(len(context2014)-common_context))/(len(context2014)))   
stat.write("No of common contexts is:  "+str(common_context)+'\n'*3)

for r1 in refs2014:
       pyr=re.findall('\d+',r1)
       for xj in pyr:
             if(len(xj)==4 and (int)(xj)>1850 and (int)(xj)<2013):
                              if(xj not in stat5year.keys()):
                                    stat5year[xj]=1
                              else:
                                    stat5year[xj]=stat5year[xj]+1
                                     



