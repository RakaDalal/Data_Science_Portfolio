import sys
reload(sys) 
sys.setdefaultencoding('UTF8')
import nltk
from nltk.collocations import *
import matplotlib.pyplot as plt
from nltk.metrics.spearman import *
'''
refs=[]
info=[]
oldrefset=[]
text=[]
abstract=[]

execfile('work25.py')
'''
trigram_measures = nltk.collocations.TrigramAssocMeasures()


#mstat1
mstat1=[]
mstat2=[]
for r in refs:
    print(refs.index(r))
    coeffs=[] 
    words2=[]
    words2=[]
    coeffs3=[]
    r[0]=str(re.sub(r'[^\w]',' ',r[0]))
    words1=nltk.word_tokenize(r[0].lower())
    finder = TrigramCollocationFinder.from_words(words1,window_size=4)
    scored2=finder.score_ngrams(trigram_measures.likelihood_ratio)
    h1=finder.nbest(trigram_measures.likelihood_ratio,4*len(words1)-7) 
    for i in range(len(text[refs.index(r)])): 
        j1=[]
        j2=[]
        te=str(re.sub(r'[^\w]',' ',text[refs.index(r)][0]))  
        words2=nltk.word_tokenize(te.lower())
        finder2=TrigramCollocationFinder.from_words(words2,window_size=4)
        scored1=finder2.score_ngrams(trigram_measures.likelihood_ratio)
        h2=finder2.nbest(trigram_measures.likelihood_ratio,4*len(words2)-7)
        coeffs2=list(set(h2).intersection(set(h1)))
        for z in coeffs2:
            for w in scored1:
                if(z in w):     
                     j1.append((z,w[1]))
            for w in scored2:
                if(z in w):
                     j2.append((z,w[1]))  
        coeffs.append(spearman_correlation(ranks_from_scores(j1),ranks_from_scores(j2)))
        coeffs3.append(len(coeffs2))
    print(max(coeffs))
    mstat1.append(max(coeffs))
    mstat2.append(max(coeffs3))
'''   
mstat3=[]
mstat4=[]
for r in refs:
    print(refs.index(r))
    coeffs=[] 
    words2=[]
    words2=[]
    coeffs3=[]
    r[0]=str(re.sub(r'[^\w]',' ',r[0]))
    words1=nltk.word_tokenize(r[0].lower())
    finder = trigramCollocationFinder.from_words(words1,window_size=3)
    scored2=finder.score_ngrams(trigram_measures.likelihood_ratio)
    h1=finder.nbest(trigram_measures.likelihood_ratio,2*len(words1)-3)  
    for i in range(len(oldrefset[refs.index(r)])):
        j1=[]
        j2=[]
        te=str(re.sub(r'[^\w]',' ',oldrefset[refs.index(r)][0]))  
        words2=nltk.word_tokenize(te.lower())
        finder2=trigramCollocationFinder.from_words(words2,window_size=3)
        scored1=finder2.score_ngrams(trigram_measures.likelihood_ratio)
        h2=finder2.nbest(trigram_measures.likelihood_ratio,2*len(words2)-3)
        coeffs2=list(set(h2).intersection(set(h1)))
        for z in coeffs2:
            for w in scored1:
                if(z in w):     
                     j1.append((z,w[1]))
            for w in scored2:
                if(z in w):
                     j2.append((z,w[1]))  
        coeffs.append(spearman_correlation(ranks_from_scores(j1),ranks_from_scores(j2)))
        coeffs3.append(len(coeffs2))
    print(max(coeffs))
    mstat3.append(max(coeffs))
    mstat4.append(max(coeffs3))
'''
mstat5=[]
mstat6=[]
for r in refs:
    print(refs.index(r))
    coeffs=[] 
    words2=[]
    words2=[]
    coeffs3=[]
    if(len(abstract[refs.index(r)])!=0): 
	    abst=str(re.sub(r'[^\w]',' ',abstract[refs.index(r)][0]))
	    words1=nltk.word_tokenize(abst.lower())
	    finder = TrigramCollocationFinder.from_words(words1,window_size=4)
	    scored2=finder.score_ngrams(trigram_measures.likelihood_ratio)
	    h1=finder.nbest(trigram_measures.likelihood_ratio,4*len(words1)-7)  
	    for i in range(len(text[refs.index(r)])):
		j1=[]
		j2=[]
		te=str(re.sub(r'[^\w]',' ',text[refs.index(r)][0]))  
		words2=nltk.word_tokenize(te.lower())
		finder2=TrigramCollocationFinder.from_words(words2,window_size=4)
		scored1=finder2.score_ngrams(trigram_measures.likelihood_ratio)
		h2=finder2.nbest(trigram_measures.likelihood_ratio,4*len(words2)-7)
		coeffs2=list(set(h2).intersection(set(h1)))
		for z in coeffs2:
		    for w in scored1:
		        if(z in w):     
		             j1.append((z,w[1]))
		    for w in scored2:
		        if(z in w):
		             j2.append((z,w[1]))  
		coeffs.append(spearman_correlation(ranks_from_scores(j1),ranks_from_scores(j2)))
		coeffs3.append(len(coeffs2))
	    print(max(coeffs))
	    mstat5.append(max(coeffs))
	    mstat6.append(max(coeffs3))


plt.hold(False)  
z=plt.hist(mstat1,bins=5,color='red')
plt.ylabel("Number of references")
plt.xlabel("Spearson correlation of trigrams(window_size=4) in title with the trigrams in text")
plt.ylim(0,max(z[0])+2)
plt.savefig("/home/raka/stats/mstat33.png")


plt.hold(False)  
z=plt.hist(mstat2,bins=8,color='red')
plt.ylabel("Number of references")
plt.xlabel("Number of trigrams in title(window_size=4) present in the text")
plt.ylim(0,max(z[0])+2)
plt.savefig("/home/raka/stats/mstat34.png")

'''
plt.hold(False)  
z=plt.hist(mstat3,bins=8,color='red')
plt.xlabel("Number of references")
plt.ylabel("Spearson correlation of trigrams in title with the trigrams in old reference set")
plt.ylim(0,max(z[0])+2)
plt.savefig("/home/raka/stats/mstat27.png")

plt.hold(False)  
z=plt.hist(mstat4,bins=8,color='red')
plt.ylabel("Number of references")
plt.xlabel("Number of trigrams in abstract present in the old reference set")
plt.ylim(0,max(z[0])+2)
plt.savefig("/home/raka/stats/mstat28.png")
'''

plt.hold(False)  
z=plt.hist(mstat5,bins=8,color='red')
plt.ylabel("Number of references")
plt.xlabel("Spearson correlation of the trigrams(window_size=4) in abstract with the trigrams in text")
plt.ylim(0,max(z[0])+2)
plt.savefig("/home/raka/stats/mstat35.png")



plt.hold(False)  
z=plt.hist(mstat6,bins=8,color='red')
plt.ylabel("Number of references")
plt.xlabel("Number of trigrams in abstract(window_size=4) present in the text")
plt.ylim(0,max(z[0])+2)
plt.savefig("/home/raka/stats/mstat36.png")







   
        
    

