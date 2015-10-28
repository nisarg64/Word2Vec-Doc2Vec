'''
Created on Jun 5, 2015

@author: Changsung Moon (csmoon2@ncsu.edu)
'''


#from bs4 import BeautifulSoup
import re
import numpy as np
#import nltk
from nltk.corpus import stopwords
from topia.termextract import extract

from gensim.models import doc2vec


def extract_keywords( doc, lower=False ):
    extractor = extract.TermExtractor()
    extractor.filter = extract.DefaultFilter()

    keywords_list = [] 
    keywords = extractor(doc)
    
    for keyword in keywords:
        if lower == True:
            keywords_list.append(keyword[0].lower())
        else:
            keywords_list.append(keyword[0])
        
    return keywords_list
                    
                    
                    
def words_to_phrases( doc ):
    keywords_list = extract_keywords( doc, lower=False )
    
    for keyword in keywords_list:
        if len(keyword.split()) > 1:
            phrase = keyword.replace(' ', '_')
            doc = doc.replace(keyword, phrase)
        
    return doc
                    
                    

def doc_to_wordlist( doc, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    #doc_text = BeautifulSoup(doc).get_text()
      
    # 2. Remove non-letters
    doc_text = re.sub("[^a-zA-Z_/@']"," ", doc)
    
    # 3. Convert words to lower case and split them
    words = doc_text.lower().split()
    
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    
    # 5. Return a list of words
    return(words)



# Define a function to split a review into parsed sentences
def doc_to_sentences( doc, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(doc.strip())
    
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call doc_to_wordlist to get a list of words
            sentences.append( doc_to_wordlist( raw_sentence, remove_stopwords ))
    
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences



# Define a function to split a doc into parsed sentences with label
def doc_to_labeled_sentences( doc, tokenizer, sent_num_start, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(doc.strip())
    
    # 2. Loop over each sentence
    sentences = []
    sent_num = sent_num_start
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call doc_to_wordlist to get a list of words
            words = doc_to_wordlist(raw_sentence, remove_stopwords)
            labeled_sentence = doc2vec.LabeledSentence(words = words, labels=['SENT_%s' % sent_num])
            sentences.append(labeled_sentence)
            sent_num = sent_num + 1
    
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return (sentences, sent_num)



def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    
    nwords = 0.
    
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec



def getAvgFeatureVecs(docs, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
     
    # Initialize a counter
    counter = 0.
    
    # Preallocate a 2D numpy array, for speed
    docFeatureVecs = np.zeros((len(docs), num_features), dtype="float32")
    
    # Loop through the documents
    for doc in docs:
        # Print a status message every 1000th document
        if counter%1000. == 0.:
            print "Document %d of %d" % (counter, len(docs))
        
        # Call the function (defined above) that makes average feature vectors
        docFeatureVecs[counter] = makeFeatureVec(doc, model, num_features)
    
        # Increment the counter
        counter = counter + 1.
        
    return docFeatureVecs



def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    
    # Return the "bag of centroids"
    return bag_of_centroids
