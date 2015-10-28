import numpy as np
import sys

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import re
from sklearn.manifold import TSNE

from func.io import convert_pdf_to_txt
from func.io import convert_docx_to_txt
from func.nlp import getAvgFeatureVecs
from func.nlp import extract_keywords
import parse_directory

reload(sys)
sys.setdefaultencoding("utf-8")


dataset_folder = "./CSC791_Corpus_Pdf/"      # For Monthly Notes dataset

num_features = 300

docs = []           # Initialize an empty list of contents
doc_names = []      # Initialize an empty list of document file names
folder_names = []

files = parse_directory.build_recursive_dir_tree(dataset_folder)

for i, file in enumerate(files):
    file_parts = str(file).split("/")
    filename = file_parts[len(file_parts)-1]
    folder = file_parts[len(file_parts)-2]
    file_path = file

    print(i)
    doc=""
    if str(file).__contains__(".txt"):
        fp = open(file,"r")
        doc = fp.read()
        doc = doc.decode("utf8")
        docs.append(doc)
        file = file.decode("utf8")
        doc_names.append(filename)
        folder_names.append(folder)
        print(folder)
    # if str(file).__contains__(".pdf"):
    #     try:
    #         doc = convert_pdf_to_txt(file_path)
    #     except:
    #         pass
    #
    #     if len(doc) > 1:
    #         doc = doc.decode("utf8")
    #         docs.append(doc)
    #         file = file.decode("utf8")
    #         doc_names.append(file)
    #         folder_names.append(folder)
    #         print(folder)
    #         fb = open(str(file)+".txt", "wb")
    #         fb.write(doc)
    #         fb.close()
    # elif str(file).__contains__(".docx"):
    #     doc = convert_docx_to_txt(file_path)
    #     if len(doc) > 1:
    #         doc = doc.decode("utf8")
    #         docs.append(doc)
    #         file = file.decode("utf8")
    #         doc_names.append(file)
    #         folder_names.append(folder)
    #         print(folder)
    #         fb = open(str(file)+".txt", "wb")
    #         fb.write(doc)
    #         fb.close()
    else:
        continue




print len(docs)
print "There are " + str(len(doc_names)) + " documents."




'''
filter_vocab_1 = [u'limited', u'four', u'facilities', u'every', u'_march', u'speci', u'second', u'implemented', u'even', u'established', u'new', 
                u'never', u'met', u'active', u'study', u'changes', u'mersive', u'actions', u'would', u'call', u'type', u'tell', u'must', u'word', 
                u'pursue', u'work', u'install', u'concepts', u'example', u'give', u'reviewing', u'want', u'end', u'provide', u'answer', u'description', 
                u'types', u'attempt', u'lost', u'maintain', u'order', u'feedback', u'office', u'fit', u'personal', u'highlights', u'expectations', u'writing', 
                u'better', u'production', u'weeks', u'combination', u'ifttt', u'classified', u'feeds', u'side', u'mean', u'hosted', u'extract', u'rd', u're', u'got', 
                u'little', u'free', u'created', u'days', u'uses', u'purpose', u'already', u'coding', u'another', u'wasn', u'top', u'needed', u'took', 
                u'likely', u'ran', u'mind', u'talking', u'manner', u'seem', u'seek', u'estimated', u'contact', u'though', u'extending', u'm', u'points', 
                u'came', u'incorporate', u'meetings', u'labels', u'proposed', u'rich', u'oct', u'di', u'de', u'dr', u'runs', u'briefed', u'mandatory', 
                u'decided', u'said', u'lots', u'away', u'future', u'hopefully', u'however', u'news', u'improve', u'suggested', u'received', 
                u'planned', u'com', u'asked', u'con', u'streck', u'anticipatory', u'trust', u'basis', u'three', u'drafting', u'much', u'interest', u'basic', u'expected', 
                u'life', u'worked', u'applied', u'aim', u'k', u'cant', u'ic', u'things', u'make', u'potentially', u'evaluate', u'several', u'july_', u'opportunity', 
                u'thoughts', u'corporate', u'left', u'identify', u'facts', u'yet', u'previous', u'adding', u'board', u'easy', u'gave', u'possible', u'possibly', 
                u'unique', u'advanced', u'cognitive', u'submission', u'specific', u'steps', u'right', u'/', u'contributing', u'everything', u'individuals', u'participation', 
                u'post', u'may_', u'adet', u'months', u'o', u'ensure', u'eight', u'efforts', u'runtime', u'automatically', u'parsing', u'mamba', u'support', u'initial', 
                u'way', u'creation', u'form', u'true', u'computing', u'evidence', u'exist', u'us', u'floor', u'nd', u'ng', u'tim', u'interested', u'papers', u'test', 
                u'welcome', u'outreach', u'and/or', u'longer', u'together', u'time', u'push', u'global', u'focus', u'comments', u'environment', u'finally', u'asking', 
                u'join', u'presented', u'leave', u'round', u'current', u'goes', u'shared', u'v', u'understanding', u'groups', u'address', u'along', u'change', u'trial', 
                u'domains', u'studies', u'useful', u'operational', u'working', u'visit', u'_stime_october_', u'live', u'scope', u'prep', u'today', u'sharing', 
                u'sessions', u'visual', u'cases', u'effort', u'capturing', u'insights', u'states', u'following', u'making', u'figure', u'december', u'agent', u'requirements', 
                u'discussion', u'write', u'product', u'may', u'applications', u'produce', u'designed', u'date', u'data', u'natural', u'outline', u'maybe', u'sp', u'st', u'q', 
                u'representation', u'talk', u'imagining', u'delivery_order_', u'years', u'course', u'still', u'vtc', u'thank', u'interesting', u'coordinated', u'main', 
                u'non', u'conversations', u'introduce', u'binks', u'half', u'discuss', u'term', u'name', u'january', u'didn', u'establishing', u'domain', u'ed', 
                u'unclassified//for', u'year', u'space', u'looking', u'attended', u'ort', u'quite', u'advance', u'discussing', u'transition', u'place', u'think', u'first', 
                u'one', u'long', u'directly', u'open', u'given', u'org/', u'indicate', u'draft', u'white', u'participated', u'national', u'mostly', u'third', u'require', 
                u'r', u'accessed', u'pre', u'say', u'note', u'potential', u'take', u'online', u'objective', u'performance', u'begin', u'sure', u'multiple', u'track', 
                u'visitors', u'especially', u'considered', u'later', u'drive', u'show', u'daily', u'written', u'going', u'awareness', u'get', u'signi', u'erent', u'geo', 
                u'regarding', u'summary', u'vision', u'ways', u'review', u'enough', u'reading', u'across', u'august', u'come', u'many', u'according', 
                u'enabling', u'among', u'r/o/i', u'overview', u'period', u'informed', u'mark', u'direction', u'enable', u'observe', u'external', u'case', u'developing', 
                u'casl', u'ongoing', u'investigating', u'engaged', u'someone', u'different', u'pat', u'etc', u'media', u'html', u'status', u'assist', u'driver', u'running', 
                u'tracking', u'without', u'coordinate', u'summer', u'rest', u'file', u'speed', u'captured', u'weekly', u'thinking', u'seems', u'setting', u'real', u'aspects', 
                u'around', u'rules', u'early', u'np', u'using', u'achieve', u'_july', u'fully', u'veracity', u'provided', u'critical', u'provides', u'measuring', u'process', 
                u'business', u'update', u'tdy', u'leadership', u'visits', u'exciting', u'went', u'_', u'ach', u'discussed', u'practical', u'ow', u'os', u'researching', 
                u'communication', u'accounts', u'determine', u'fast', u'area', u'start', u'low', u'lot', u'complete', u'trying', u'october', u'detailed', u'certain', 
                u'describe', u'deep', u'general', u'aw', u'personnel', u'storage', u'requested', u'students', u'includes', u'important', u'included', u'building', u'invest', 
                u'u', u'starting', u'represent', u'consider', u'lack', u'month', u'follow', u'decisions', u'reasoning', u'content', u'opportunities', u'th', u'presentation', 
                u'activities', u'fall', u'samsi', u'leads', u'list', u'joined', u'large', u'small', u'past', u'design', u'perspective', u'pass', u'investment', u'sub', 
                u'brief', u'version', u'learned', u'full', u'component', u'november', u'search', u'allows', u'experience', u'prior', 
                u'amount', u'social', u'action', u'via', u'put', u'semester', u'establish', u'eye', u'objectives', u'two', u'company', u'particular', u'known', u'science', 
                u'resources', u'learn', u'scramble', u'history', u'_description', u'suggestions', u'ecting', u'cation', u'sense', 
                u'needs', u'rather', u'ft', u'tried', u'fy', u'reflect', u'_april', u'coming', u'develop', u'response', u'short', u'documentation', u'playing', u'help', 
                u'september', u'developed', u'soon', u'installed', u'held', u'paper', u'solving', u'actually', u'late', u'systems', u'might', 
                u'motivated', u'always', u'level', u'capability', u'courses', u'found', u'week', u'everyone', u'mental', u'generation', u'house', u'energy', u'hard', u'idea', 
                u'engaging', u'expect', u'beyond', u'really', u'funding', u'since', u'research', u'participants', u'evaluation', u'base', u'imagine', 
                u'ask', u'beginning', u'generate', u'english', u'introduction', u'conducted', u'_june', u'feed', u'major', u'feel', u'number', u'done', u'story', u'iarpa', 
                u'interact', u'least', u'behind', u'part', u'focusing', u'believe', u'kind', u'b', u'sta', u'toward', u'built', u'self', u'client', u'also', u'internal', 
                u'build', u'play', u'towards', u'serve', u'sets', u'plan', u'significant', u'approved', u'clear', u'km', u'chris_e', u'particularly', u'session', u'carefully', 
                u'find', u'impact', u'distributed', u'pretty', u'preparing', u'closely', u'investigate', u'common', u'x', u'wrote', u'set', u'art', 
                u'ara', u'see', u'individual', u'close', u'preparation', u'visiting', u'currently', u'please', u'won', u'various', u'probably', u'numerous', u'available', 
                u'recently', u'creating', u'initially', u'attention', u'mentoring', u'interface', u'c', u'last', u'_april_', u'annual', u'foreign', u'connection', u'let', 
                u'point', u'simple', u'throughout', u'described', u'monthly', u'create', u'due', u'meeting', u'pm', u'generating', u'miscellaneous', u'understand', u'march_', 
                u'look', u'reliable', u'bill', u'receive', u'evaluating', u'formats', u'guide', u'scheduled', u'hoping', u'ready', u'read', u'grand', u'higher', 
                u'used', u'moment', u'levels', u'moving', u'user', u'identi', u'consulted', u'recent', u'task', u'spent', u'person', u'y', u'spend', u'judgments', u'reviewed', 
                u'questions', u'world', u'informal', u'source', u'location', u'input', u'excited', u'march', u'format', u'big', u'couple', u'bit', u'formal', u'd', u'semi', 
                u'systematically', u'identifying', u'communications', u'continue', u'back', u'added', u'examples', u'curious', u'delivered', u'scale', u'describing', 
                u'decision', u'per', u'meade', u'either', u'run', u'processing', u'continuing', u'deliverables', u'step', u'anything', u'range', u'seeing', u'computational', 
                u'within', u'appropriate', u'spending', u'apparently', u'question', u'specifically', u'forward', u'invite', u'properly', u'line', 
                u'_may', u'planet', u'similar', u'called', u'associated', u'defined', u'doesn', u'single', u'diverse', u'department', u'elements', u'users', u'problems', 
                u'prepared', u'helping', u'generated', u'adelsperger', u'structure', u'e', u'required', 
                u'far', u'code', u'o/i', u'results', u'existing', u'edu', u'go', u'issues', u'seemed', u'send', u'quarterly', u'include', u'sent', u'continues', u'continued', 
                u'categories',  u'prototypes', u'try', u'etime', u'access', u'led', u'exchange', 
                u'leveraging', u'delivering', u'explore', u'focused', u'others', u'safer', u'great', u'engage', u'involved', u'products', u'implement', u'manage', u'outcomes', 
                u'apply', u'use', u'feb', u'next', u'upcoming', u'started', u'integrate', u'customer', u'account', u'f', 
                u'meet', u'invoices', u'high', u'bene', u'something', u'unclassified', u'attend', u'allow', u'move', u'including', u'looks', u'mentioned', u'safer_planet', u'll', 
                u'defining', u'greater', u'auto', u'outlined', u'day', u'_years', u'february', u'identified', u'related', 
                u'measure', u'integrated', u'activity', u'attending', u'completely', u'ect', u'lts', u'interaction', u'timeglider', u'g', u'determining', u'could', u'keep', 
                u'facilitate', u'improving', u'quality', u'management', u'system', u'final', u'completed', u'slides', 
                u'rst', u'july', u'haven', u'culture', u'submitted', u'briefing', u'visited', u'providing', u'april_', u'need', u'_january', u'documents', u'inform', u'agency', 
                u'able', u'mid', u'mechanism', u'_kickoff', u'incorporating', u'subject', u'rigor', 
                u'gather', u'face', u'gain', u'text', u'agreed', u'bring', u'planning', u'based', u'jan', u'local', u'hope', u'insight', u'means', u'familiar', u'overall', 
                u'areas', u'processes', u'h', u'view', u'j', u'frame', u'computer', u'signalscape', u'attempting', 
                u'nature', u'state', u'progress', u'email', u'ability', u'importance', u'deliver', u'implementing', u'job', u'key', u'figure_', u'approval', u'taking', 
                u'presentations', u'april', u'relevant', u'co', u'tuesday', u'environments', u'respect', u'addition', u'define', u'diversity', 
                u'finished', u'hmg', u'present', u'multi', u'value', u've', u'almost', u'thus', u'site', u'helped', u'vs', u'capture', u'perhaps', u'began', u'cross', u'difficult', 
                u'http', u'reporting', u'upon', u'automation', u'student', u'//', u'center', u'well', u'thought', u'patterns', u'position', 
                u'latest', u'less', u'sources', u'happiness', u'capabilities', u'add', u'usage', u'stakeholders', u'know', u'necessary', u'like', u'become', u'works', u'page', 
                u'library', u'lead', u'broad', u'although', u'actual', u'carried', u'getting', u'@ncsu', 
                u'introduced', u'previously', u'assess', u'quickly', u'additional', u'transfer', u'north', u'funds', u'hq', u'delivery', u'hi', u'automated', u'made', u'whether', 
                u'official', u'problem', u'piece', u'display', u'u//fouo', u'contribute', u'ing', u'functions', u'variety', u'detail', u'virtual', u'details', u'scif', u'june', u'sensemaking' ]

'''



model = Word2Vec.load("./CSC791_Corpus_300features_10minwords_10context_FalseStopwords_Phrase")
#model2 = Doc2Vec.load("./CSC791_Corpus_doc2vec_model")

vectors_origin = model.syn0     # word vector for each vocabulary
vectors = []

# num_clusters = vectors_origin.shape[0] / 100
#
# print(num_clusters)
#
# kmeans_clustering = KMeans( n_clusters = num_clusters )
# idx = kmeans_clustering.fit_predict( vectors_origin )
#
# word_centroid_map = dict(zip( model.index2word, idx ))
#
# # For the first 10 clusters
# for cluster in xrange(0,num_clusters):
#     #
#     # Print the cluster number
#     print "\nCluster %d" % cluster
#     #
#     # Find all of the words for that cluster number, and print them out
#     words = []
#     for i in xrange(0,len(word_centroid_map.values())):
#         if( word_centroid_map.values()[i] == cluster ):
#             words.append(word_centroid_map.keys()[i])
#     print words



vocab_list_origin = model.vocab.keys()  # The list of the model's vocabulary
vocab_list = []

stops = set(stopwords.words("english"))


vecs = []
# Extract non-stopwords from vocab_list and vectors
for i in range(0, len(vocab_list_origin)):
    vocab = vocab_list_origin[i]

    if vocab not in stops and len(vocab) > 3:
        vocab_list.append(vocab)
        vecs.append(model[vocab].reshape((1, num_features)))



vecs = np.concatenate(vecs)
vectors = np.array(vecs, dtype='float') #TSNE expects float type values


print "There are " + str(len(vectors_origin)) + " original vocabularies."
print "There are " + str(len(vocab_list)) + " vocabularies."
print "There are " + str(len(vectors)) + " vocabulary vectors."

import csv
# with open('word-vecs.csv', 'wb') as csvfile:
#     writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     for j in range(0,len(vocab_list)):
#         writer.writerow([vocab_list[j]+","+",".join([str(i) for i in vectors[j]])])

# Extract the list of keywords for each document
clean_docs = []
clean_doc_names = []
clean_folder_names = []

for i in range(0, (len(docs)-1)):
    doc = docs[i]
    keywords_for_doc_vec = []
    keywords_list = extract_keywords( doc, lower=True )

    for keyword in keywords_list:
        if len(keyword.split()) > 1:    # If the keyword is a phrase, replace white spaces with '_'
            keyword = keyword.replace(' ', '_')

        if keyword in vocab_list:
            keywords_for_doc_vec.append(keyword)

    if len(keywords_for_doc_vec) > 0:
        clean_docs.append(keywords_for_doc_vec)
        clean_doc_names.append(doc_names[i])
        clean_folder_names.append(folder_names[i])

# Average vectors for all of words for document vector
doc_vecs_tmp = getAvgFeatureVecs( clean_docs, model, num_features )

print(clean_folder_names)

vecs = []
# Extract non-stopwords from vocab_list and vectors
for i in range(0, len(doc_vecs_tmp)):
    doc_vec = doc_vecs_tmp[i]
    vecs.append(doc_vec.reshape((1, num_features)))

vecs = np.concatenate(vecs)
doc_vecs = np.array(vecs, dtype='float') #TSNE expects float type values

print len(doc_vecs)

#print(re.sub('[^a-zA-Z0-9-.@_/]', '', str(clean_doc_names[j]))+","+",".join([str(i) for i in doc_vecs[0]]))
#
# with open('csc791-doc-vecs.csv', 'wb') as csvfile:
#     writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     for j in range(0,len(doc_vecs)):
#         writer.writerow([re.sub('[^a-zA-Z0-9-.@_/]', '', str(clean_doc_names[j]))+","+",".join([str(i) for i in doc_vecs[j]])])


n_components = 2    # 2 for 2D, and 3 for 3D
#
model_tsne = TSNE(n_components, random_state=0)
# words_tsne = model_tsne.fit_transform(vectors)
docs_tsne = model_tsne.fit_transform(doc_vecs)

# words_docs_tsne = np.concatenate((words_tsne, docs_tsne))
labels = np.concatenate((vocab_list, clean_doc_names))

# 2D Visualization

# # x1 and y1 are for keywords with blue dots
# x1 = words_tsne[:,0]
# y1 = words_tsne[:,1]
#
# fig, ax = plt.subplots()
#
# ax.scatter(x1, y1, color='blue')
#
# for i, txt in enumerate(vocab_list):
#     ax.annotate(txt, (x1[i],y1[i]))
#

fig, ax = plt.subplots()
# x1 and y1 are for documents with red dots
x2 = docs_tsne[:,0]
y2 = docs_tsne[:,1]

ax.scatter(x2, y2, color='red')

for i, txt in enumerate(clean_folder_names):
    ax.annotate(txt, (x2[i],y2[i]))


fig, ax = plt.subplots()
# x1 and y1 are for documents with red dots
x2 = docs_tsne[:,0]
y2 = docs_tsne[:,1]

ax.scatter(x2, y2, color='red')

for i, txt in enumerate(clean_doc_names):
    ax.annotate(txt, (x2[i],y2[i]))


plt.show()


'''
# 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# x1, y1 and z1 are for keywords with blue dots
x1 = words_tsne[:,0]
y1 = words_tsne[:,1]
z1 = words_tsne[:,2]

ax.scatter(x1, y1, z1, color='blue', marker='o')

# x2, y2 and z2 are for documents with red dots
x2 = docs_tsne[:,0]
y2 = docs_tsne[:,1]
z2 = docs_tsne[:,2]

ax.scatter(x2, y2, z2, color='red', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
'''


