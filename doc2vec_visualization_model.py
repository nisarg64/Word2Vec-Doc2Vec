from gensim.models import doc2vec
import parse_directory
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys


reload(sys)
sys.setdefaultencoding("utf-8")

dataset_folder = "./CSC791_Corpus_Pdf/"

docnames = []
folders = []
docLabels = []
docs = files = parse_directory.build_recursive_dir_tree(dataset_folder)

for doc in docs:
    if str(doc).endswith(".txt"):
        file_parts = str(doc).split("/")
        filename = file_parts[len(file_parts)-1]
        folder = file_parts[2]
        docLabels.append(doc)
        docnames.append(filename)
        folders.append(folder)

print(len(docLabels))
print(folders)

model = doc2vec.Doc2Vec.load("doc2vec.model")

num_features = 300

print(len(model.docvecs))

vecs = []
# Extract non-stopwords from vocab_list and vectors
for doc in docLabels:
    doc_vec = model.docvecs[doc]
    vecs.append(doc_vec.reshape((1, num_features)))

vecs = np.concatenate(vecs)
doc_vecs = np.array(vecs, dtype='float') #TSNE expects float type values

n_components = 2    # 2 for 2D, and 3 for 3D
#
model_tsne = TSNE(n_components, random_state=0)

docs_tsne = model_tsne.fit_transform(doc_vecs)

fig, ax = plt.subplots()
# x1 and y1 are for documents with red dots
x2 = docs_tsne[:,0]
y2 = docs_tsne[:,1]

ax.scatter(x2, y2, color='red')

for i, txt in enumerate(folders):
    ax.annotate(txt, (x2[i],y2[i]))


plt.show()


