import pickle
import gzip
from scipy.sparse import hstack
from keras.models import Model, load_model
import sys




def classification(classifier, dic, text):
	'''Args:
		classifier: The best model trained on the entire data
		dic: A dictionary storing the Vocabulary, word level tf-idf object, ngram level tf-idf object, characters level tf-idf object and label encoder
		text: input text whose label is to be predicted
	Returns:
		Predicted Label of the input_text'''



	Vocabulary=dic["Vocabulary"]
	tfidf_vect=dic["tfidf_vect"] # word level tf-idf	
	tfidf_vect_ngram=dic["tfidf_vect_ngram"] # ngram level tf-idf
	tfidf_vect_ngram_chars=dic["tfidf_vect_ngram_chars"] # characters level tf-idf
	encoder=dic["encoder"] #label encoder
	# replacing out of vocabulary words with OOV
	for word in text.split():
		if word not in Vocabulary:
			text=text.replace(word, "OOV")
	text=[text]
	x_tfidf =  tfidf_vect.transform(text)		
	x_tfidf_ngram =  tfidf_vect_ngram.transform(text)
	x_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(text) 
	# Combining Word Level TF IDF Vectors, Ngram Level TF IDF Vectors and Character Level TF IDF Vectors
	X=hstack([x_tfidf, x_tfidf_ngram, x_tfidf_ngram_chars]).toarray()
	Y=classifier.predict(X)
	Y=Y.argmax(axis=-1)
	Y=encoder.inverse_transform(Y)
	return (Y[0])




def main():
	text_for_prediction=sys.argv[1]
	# reading the files
	path_to_model="./"
	f = gzip.open(path_to_model+'dictionary.pklz','rb')
	dic = pickle.load(f)
	f.close()
	classifier = load_model('my_trained_model.h5')
	print ("reading done")
	print ("The label:\t"+classification(classifier, dic, text_for_prediction))

if __name__=='__main__' :
    main()


