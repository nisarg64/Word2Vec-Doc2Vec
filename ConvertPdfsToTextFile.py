import logging
import datetime
import os
import nltk

from os import listdir
from os.path import isfile, isdir, join, splitext

from func.io import convert_pdf_to_txt
from func.io import convert_docx_to_txt
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
        doc = ""

        if file_extension == ".pdf":
            doc = convert_pdf_to_txt(file_path)
        elif file_extension == ".docx":
            doc = convert_docx_to_txt(file_path)
        else:
            continue

        if doc != "":
            doc_sentence = []
            doc = doc.decode("utf8").encode('ascii', 'ignore').decode('ascii')
            doc = words_to_phrases(doc)
            doc = doc.lower()
            sentences += doc_to_sentences(doc, tokenizer, remove_stopwords=True)
            doc_sentence = doc_to_sentences(doc, tokenizer, remove_stopwords=True)
            print(file_path)
            file1 = file.split("_")
            month = file1[2].split(".")[0]+"-"+file1[1]
            author = file1[0]
            for item in doc_sentence:
                item = " ".join(item)
                #sentences_string += "\n"+item+" | "+author+" | "+file.decode("utf8").encode('ascii', 'ignore').decode('ascii')+" | "+month


print(sentences_string.count("\n"))
with open("sentences.txt", "w") as text_file:
    text_file.write("%s" % sentences_string)