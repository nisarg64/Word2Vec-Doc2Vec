import codecs
import logging
from os import listdir
from os.path import isfile, isdir, join, splitext

import nltk
from func.nlp import doc_to_sentences
from func.nlp import words_to_phrases

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dataset_folder = "./CSC791_Corpus_RawTxt/"     # For Monthly Notes dataset


# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []      # Initialize an empty list of sentences
file_names = []
sentences_string = ""
input_folders = [ sub_dir for sub_dir in listdir(dataset_folder) if isdir(join(dataset_folder, sub_dir)) ]

for folder in input_folders:
    dir_path = dataset_folder + folder + str("/")
    files = [ f for f in listdir(dir_path) if isfile(join(dir_path,f)) ]
    for file in files:
        file_path = dir_path + file
        file_name, file_extension = splitext(file_path)
        doc = codecs.open(file_path,"r",encoding='utf-8', errors='ignore').read()
        print(file)
        if doc != "":
            doc_sentence = []
            doc = doc
            doc = words_to_phrases(doc)
            doc = doc.lower()
            sentences += doc_to_sentences(doc, tokenizer, remove_stopwords=True)
            doc_sentence = doc_to_sentences(doc, tokenizer, remove_stopwords=True)
            for item in doc_sentence:
                item = " ".join(item)
                if(item != ""  and len(item) > 20):
                    sentences_string += "\n"+item+" | "+file.decode("utf8").encode('ascii', 'ignore').decode('ascii')


print(sentences_string.count("\n"))
with open("csc791-sentences.txt", "w") as text_file:
    text_file.write("%s" % sentences_string)

import random
def select_random_sentences(n):
    lines = []
    with open("csc791-sentences.txt", "r") as f:
        lines = f.readlines();
    with open("csc791-sentences-random.txt", "w") as text_file:
        for i in range(0,n):
            random_line_num = random.randrange(0, len(lines))
            text_file.write("%s\n" % lines[random_line_num])


select_random_sentences(5500)
