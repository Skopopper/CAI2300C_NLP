import time, json, logging
from gensim.models import Word2Vec
logging.basicConfig(filename='train.log',level=logging.INFO,format='%(asctime)s %(message)s')
from gensim.models.word2vec import Text8Corpus
corpus=Text8Corpus('text8')
t=time.time()
m=Word2Vec(sentences=corpus,vector_size=100,window=5,min_count=5,sg=1,negative=5,epochs=5,workers=2,seed=42)
m.save('text8_w2v.model')
json.dump({'seconds':round(time.time()-t),'vocab':len(m.wv)},open('done.json','w'))
