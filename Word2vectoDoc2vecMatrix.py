import csv
import numpy as np

word_vectors=[]
word_list=[]
with open('word-vecs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for i,row in enumerate(reader):
        vector=row[0].split(",")
        word_list.append(vector[0])
        del vector[0]
        word_vectors.append(vector)

doc_vectors=[]
doc_list=[]
with open('doc-vecs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        vector=row[0].split(",")
        doc_list.append(vector[0])
        del vector[0]
        doc_vectors.append(vector)


word_vector_matrix = np.matrix(np.array(word_vectors).astype('float'))
doc_vector_matrix = np.matrix(np.array(doc_vectors).astype('float'))
doc_vector_matrix_transpose = doc_vector_matrix.T

print(word_vector_matrix.shape)
print(doc_vector_matrix_transpose.shape)
print(len(word_list))
print(len(doc_list))

mat = np.dot(word_vector_matrix, doc_vector_matrix_transpose)
print(mat.shape)

mat_array = np.asarray(mat)
rows = len(mat_array)
print(rows)
print(len(mat_array[0]))

with open('word2doc.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(0,rows):
        vector = mat_array[i]
        writer.writerow(vector)


