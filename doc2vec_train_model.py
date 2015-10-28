
import gensim.models.doc2vec as Doc2Vec
import parse_directory

__author__ = 'nisarg'



from os import listdir
from os.path import isfile, join

dataset_folder = "./CSC791_Corpus_Pdf/"

docLabels = []
docs = files = parse_directory.build_recursive_dir_tree(dataset_folder)


data = []
for doc in docs:
    if str(doc).endswith(".txt"):
        docLabels.append(doc)
        f=open(doc, 'r')
        data.append(f.read())
        f.close()
    else:
        pass

print(docLabels)

class DocIterator(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            labels=[self.labels_list[idx]]
            words=doc.split()
            yield Doc2Vec.LabeledSentence(words,labels)


it = DocIterator(data, docLabels)

model = Doc2Vec.Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(it)
for epoch in range(10):
    model.train(it)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(it)

model.save("doc2vec.model")