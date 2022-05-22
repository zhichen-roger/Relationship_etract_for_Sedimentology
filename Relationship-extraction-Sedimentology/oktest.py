import numpy as np
print("============================")

print('reading word embedding data...')
vec = []
word2id = {}
# 词向量
f = open('./origin_data/glove.6B.100d.txt', encoding='utf-8')
content = f.readline()
content = content.strip().split()
dim = int(content[1])
while True:
    content = f.readline()
    if content == '':
        break
    content = content.strip().split()
    word2id[content[0]] = len(word2id)
    content = content[1:]
    content = [(float)(i) for i in content]
    vec.append(content)
f.close()
word2id['UNK'] = len(word2id)
word2id['BLANK'] = len(word2id)

vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
vec = np.array(vec, dtype=np.float32)
print(vec)

print("============================")

print('reading relation to id')
relation2id = {}
f = open('./origin_data/relation2idtest.txt', 'r', encoding='utf-8')
while True:
    content = f.readline()
    if content == '':
        break
    content = content.strip().split(' | ')
    relation2id[content[0]] = int(content[1])
f.close()
print(relation2id)