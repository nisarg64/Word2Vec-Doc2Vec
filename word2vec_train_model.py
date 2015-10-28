from io import StringIO

import logging
import nltk

from os import listdir
from os.path import isfile, isdir, join, splitext
from gensim.models import word2vec

from func.io import convert_pdf_to_txt
from func.io import convert_docx_to_txt
from func.nlp import doc_to_sentences
from func.nlp import words_to_phrases
import parse_directory

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dataset_folder = "./CSC791_Corpus_Pdf/"     # For Monthly Notes dataset



files = parse_directory.build_recursive_dir_tree(dataset_folder)

print(len(files))

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []      # Initialize an empty list of sentences


for i,file in enumerate(files):
    file_parts = str(file).split("/")
    filename = file_parts[len(file_parts)-1]
    doc = ""
    print(i)
    try:
        if str(file).__contains__(".txt"):
            fp = open(file,"r")
            doc = fp.read()
        # elif str(file).__contains__(".pdf"):
        #     doc = convert_pdf_to_txt(file)
        # elif str(file).__contains__(".docx"):
        #     doc = convert_docx_to_txt(file)
        else:
            continue
    except:
        continue

    if doc != "":
        doc = doc.decode("utf8")
        doc = words_to_phrases(doc)
        doc = doc.lower()


        sentences += doc_to_sentences(doc, tokenizer, remove_stopwords=False)


print len(sentences)
with open("csc791_sentences.txt", "w") as text_file:
    text_file.write("%s" % sentences)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 10   # Minimum word count
num_workers = 2      # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)
# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "CSC791_Corpus_300features_10minwords_10context_FalseStopwords_Phrase"
model.save(model_name)



