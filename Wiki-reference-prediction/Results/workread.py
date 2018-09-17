import nltk
import sys
from nltk.collocations import *
import matplotlib.pyplot as plt
from nltk.metrics.spearman import *
import random
from collections import Counter
import math
import csv
import re
#import workdone
#import workreadexp
#from workdone import f_workdone
#from workreadexp import f_workreadexp
import linecache

def f_workread(text,site,bsite,esite,l):
        
        d=open("./rank"+str(bsite)+"_"+str(esite)+".txt",'w')  
	reload(sys) 
	sys.setdefaultencoding('UTF8')
	bigram_measures = nltk.collocations.BigramAssocMeasures()
	trigram_measures = nltk.collocations.TrigramAssocMeasures()
	bloblist=list(set(text))
	prec=[]
	pos_tag=[]
	scores=[]
	abst_bigrams2=[]
	abst_trigrams3=[]
	scores_tri=[]
	scores_tri4=[]
	abst_trigrams4=[]
	scores_bi3=[]
	abst_bigrams3=[]
	iddf=[]
	
        reload(sys) 
	sys.setdefaultencoding('UTF8')
	bigram_measures = nltk.collocations.BigramAssocMeasures()
	trigram_measures = nltk.collocations.TrigramAssocMeasures()
	bloblist=list(set(text))
	for x in range(len(text)):
	    text[x]=str(re.sub(r'[^\w]',' ',text[x]))
	    text[x]=text[x].lower() 


        for i in range(bsite-1,esite):
	    ss=i-bsite+1
            prec.append([])
	    for j in range(15):
		prec[ss].append([])

	with open("pos_tag.csv", "r") as f:
	    reader = csv.reader(f)
            jjj=0
	    for row in reader:
                for i in range(bsite-1,esite):
	            ss=i-bsite+1
                    words2=[]
                    print row[0]
                    zz=row[0].split("\",\"")
                    for kk in range(len(zz)):
                        zz[kk]=eval(zz[kk])
		    if(len(zz)!=0):
		        xx=[w[0] for w in zz if w[1].startswith('N')]
		        for word in xx:
		           words2.append(text[i].count(word))
		        if(len(words2)==0):
		           continue
		        if(max(words2)!=0):
		           prec[ss][0].append((str(l[jjj]),max(words2)))
		        k=0
		        for word in xx:
		           if(text[i].count(word)!=0):
		                 k=k+1 
		        prec[ss][3].append((str(l[jjj]),k))  
                        words2=[]
		        xx=[w[0] for w in zz if w[1].startswith('V')]
		        for word in xx:
		           words2.append(text[i].count(word))
		        if(len(words2)==0):
		           continue 
		        if(max(words2)!=0):
		           prec[ss][1].append((str(l[jjj]),max(words2)))
		        k=0
		        for word in xx:
		           if(text[i].count(word)!=0):
		               k=k+1 
		        prec[ss][4].append((str(l[jjj]),k))
                        words2=[]
		        xx=[w[0] for w in zz if (not w[1].startswith('V') and not w[1].startswith('N'))]
		        for word in xx:
		           words2.append(text[i].count(word))
		        if(len(words2)==0):
		           continue
		        if(max(words2)!=0):
		           prec[ss][2].append((str(l[jjj]),max(words2)))
	          	k=0
		        for word in xx:
		           if(text[i].count(word)!=0):
		               k=k+1 
		        prec[ss][5].append((str(l[jjj]),k))
                        for j in range(6):
		           prec[ss][j].sort(key=lambda tup: tup[1])
		           if(len(prec[ss][j])>50):
		                 prec[ss][j]=prec[ss][j][:50] 
                        d.write('\n'*2+"***"+"SITE:-"+str(site[i])+'\n'*2) 
            jjj=jjj+1   
 
	with open("scores.csv", "r") as f,open("abst_bigrams2.csv", "r") as f2:
	    reader1 = csv.reader(f)
            reader2 = csv.reader(f2)
            jjj=0  
	    for row1,row2 in zip(reader1,reader2):
                for i in range(bsite-1,esite):
	            ss=i-bsite+1
		    scores=row1.split("\",\"")
                    grams=row2.split("\",\"")
                    for kk in range(len(scores)):
                       scores[kk]=eval(scores[kk])
                    for kk in range(len(grams)):
                       grams[kk]=eval(grams[kk])
                    k = eval(linecache.getline("scoresexp.csv",i))
                    scored2=[]
                    for p in k:
                        scored2.append(eval(p))
                    k = eval(linecache.getline("abst_bigrams2exp.csv",i))
                    h2=[]
                    for p in k:
                        h2.append(eval(p))   
		    coeffs2=list(set(h2).intersection(set(grams)))
		    j1=[]
                    j2=[]
                    for z in coeffs2:
	                for w in scored2:
			    if(z in w):     
		                j1.append((z,w[1]))
		        for w in scores:
			    if(z in w):
		                j2.append((z,w[1]))  
		    prec[ss][6].append((str(l[jjj]),spearman_correlation(ranks_from_scores(j1),ranks_from_scores(j2))))
		    prec[ss][7].append((str(l[jjj]),len(coeffs2)))
                    for j in range(6,8):
		        prec[ss][j].sort(key=lambda tup: tup[1])
		        if(len(prec[ss][j])>50):
		            prec[ss][j]=prec[ss][j][:50]
                #scores.append(row)
                jjj=jjj+1
	'''       
	with open("abst_bigrams2.csv", "r") as f:
	    reader = csv.reader(f)
	    for row in reader:
		abst_bigrams2.append(row)
        ''' 

	with open("scores_bi3.csv", "r") as f,open("abst_bigrams3.csv", "r") as f2:
            reader1 = csv.reader(f)
            reader2 = csv.reader(f2)
            jjj=0  
	    for row1,row2 in zip(reader1,reader2):
                for i in range(bsite-1,esite):
	                ss=i-bsite+1 
			scores=row1.split("\",\"")
		        grams=row2.split("\",\"")
		        for kk in range(len(scores)):
		            scores[kk]=eval(scores[kk])
		        for kk in range(len(grams)):
		            grams[kk]=eval(grams[kk])
		        k = eval(linecache.getline("scores_bi3exp.csv",i))
                        scored2=[]
                        for p in k:
                            scored2.append(eval(p))
                        k = eval(linecache.getline("abst_bigrams3exp.csv",i))
                        h2=[]
                        for p in k:
                            h2.append(eval(p))         
			coeffs2=list(set(h2).intersection(set(grams)))
			j1=[]
		        j2=[]
		        for z in coeffs2:
			    for w in scored2:
				if(z in w):     
				    j1.append((z,w[1]))
			    for w in scores:
				if(z in w):
				    j2.append((z,w[1]))  
			prec[ss][8].append((str(l[jjj]),spearman_correlation(ranks_from_scores(j1),ranks_from_scores(j2))))
			prec[ss][9].append((str(l[jjj]),len(coeffs2)))
		        for j in range(8,10):
			    prec[ss][j].sort(key=lambda tup: tup[1])
			    if(len(prec[ss][j])>50):
				prec[ss][j]=prec[ss][j][:50] 
               	jjj=jjj+1    

        '''
	with open("abst_bigrams3.csv", "r") as f:
	    reader = csv.reader(f)
	    for row in reader:
		abst_bigrams3.append(row)
    
        
	with open("abst_trigrams3.csv", "r") as f:
	    reader = csv.reader(f)
	    for row in reader:
		abst_trigrams3.append(row)

	with open("abst_trigrams4.csv", "r") as f:
	    reader = csv.reader(f)
	    for row in reader:
		abst_trigrams4.append(row)
       ''' 
       
	with open("scores_tri.csv", "r") as f,open("abst_trigrams3.csv", "r") as f2:
	    reader1 = csv.reader(f)
            reader2 = csv.reader(f2)
            jjj=0  
	    for row1,row2 in zip(reader1,reader2):
                for i in range(bsite-1,esite):
	                ss=i-bsite+1
			scores=row1.split("\",\"")
		        grams=row2.split("\",\"")
		        for kk in range(len(scores)):
		            scores[kk]=eval(scores[kk])
		        for kk in range(len(grams)):
		            grams[kk]=eval(grams[kk])
                        k = eval(linecache.getline("scores_triexp.csv",i))
                        scored2=[]
                        for p in k:
                            scored2.append(eval(p))
                        k = eval(linecache.getline("abst_trigrams3exp.csv",i))
                        h2=[]
                        for p in k:
                            h2.append(eval(p))  
			coeffs2=list(set(h2).intersection(set(grams)))
			j1=[]
		        j2=[]
		        for z in coeffs2:
			    for w in scored2:
				if(z in w):     
				    j1.append((z,w[1]))
			    for w in scores:
				if(z in w):
				    j2.append((z,w[1]))  
			prec[ss][10].append((str(l[jjj]),spearman_correlation(ranks_from_scores(j1),ranks_from_scores(j2))))
			prec[ss][11].append((str(l[jjj]),len(coeffs2)))
		        for j in range(10,12):
			    prec[ss][j].sort(key=lambda tup: tup[1])
			    if(len(prec[ss][j])>50):
				prec[ss][j]=prec[ss][j][:50]
                jjj=jjj+1 

	with open("scores_tri4.csv", "r") as f,open("abst_trigrams4.csv", "r") as f2:
	    reader1 = csv.reader(f)
            reader2 = csv.reader(f2)
            jjj=0   
	    for row1,row2 in zip(reader1,reader2):
		for i in range(bsite-1,esite):
	                ss=i-bsite+1
		        scores=row1.split("\",\"")
		        grams=row2.split("\",\"")
		        for kk in range(len(scores)):
		            scores[kk]=eval(scores[kk])
		        for kk in range(len(grams)):
		            grams[kk]=eval(grams[kk])
                        k = eval(linecache.getline("scores_tri4exp.csv",i))
                        scored2=[]
                        for p in k:
                            scored2.append(eval(p))
                        k = eval(linecache.getline("abst_trigrams4.csv",i))
                        h2=[]
                        for p in k:
                            h2.append(eval(p))
                        coeffs2=list(set(h2).intersection(set(grams)))  
			j1=[]
		        j2=[]
		        for z in coeffs2:
			    for w in scored2:
				if(z in w):     
				    j1.append((z,w[1]))
			    for w in scores:
				if(z in w):
				    j2.append((z,w[1]))  
			prec[ss][12].append((str(l[jjj]),spearman_correlation(ranks_from_scores(j1),ranks_from_scores(j2))))
			prec[ss][13].append((str(l[jjj]),len(coeffs2))) 
		        for j in range(12,14):
			    prec[ss][j].sort(key=lambda tup: tup[1])
			    if(len(prec[ss][j])>50):
				prec[ss][j]=prec[ss][j][:50]
                jjj=jjj+1
	

        for i in range(bsite-1,esite):
	    ss=i-bsite+1 
	    d.write('\n'*2+"***"+"SITE:-"+str(site[i])+'\n'*2) 
	    for j in range(15):
		d.write('\n'*2+"FEATURENO:-"+str(j)+'\n'*2)
		for k in range(len(prec[ss][j])):  
		    d.write("AA"+str(prec[ss][j][k]))
		    d.write('\n') 
	

        print "workread done"
	#f_workreadexp(text,site,bsite,esite,abst,l,pos_tag,scores,abst_bigrams2,abst_trigrams3,scores_tri,scores_tri4,abst_trigrams4,scores_bi3,abst_bigrams3,iddf)
	#f_workdone(text,site,bsite,esite,prec,abst,pos_tag,scores,abst_bigrams2,abst_trigrams3,scores_tri,scores_tri4,abst_trigrams4,scores_bi3,abst_bigrams3,iddf,l)

