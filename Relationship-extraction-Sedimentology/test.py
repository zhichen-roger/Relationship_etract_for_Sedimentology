import numpy as np
import numpy as np
print("============================")

print('reading word embedding data...')
vec = []
word2id = {}
# 词向量
f = open('./origin_data/vec.txt', encoding='utf-8')
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
print("============================")
# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122

# find the index of x in y, if x not in y, return -1
def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag

# length of sentence is 70
fixlen = 100
# max length of position embedding is 60 (-60~+60)
maxlen = 60

train_sen = {}  # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
train_ans = {}  # {entity pair:[label1,label2,...]} the label is one-hot vector

print('reading train data...')
f = open('./origin_data/traintest.txt', 'r', encoding='utf-8')

while True:
    content = f.readline()
    if content == '':
        break

    content = content.strip().split(' | ')
    # get entity name
    en1 = content[0]
    en2 = content[1]
    print("en1: " + en1)
    print("en2: " + en2)
    relation = 0
    if content[2] not in relation2id:
        relation = relation2id['NA']
    else:
        relation = relation2id[content[2]]
    # put the same entity pair sentences into a dict
    tup = (en1, en2)
    label_tag = 0
    if tup not in train_sen:
        train_sen[tup] = []
        train_sen[tup].append([])
        y_id = relation
        label_tag = 0
        label = [0 for i in range(len(relation2id))]
        label[y_id] = 1
        train_ans[tup] = []
        train_ans[tup].append(label)
    else:
        y_id = relation
        label_tag = 0
        label = [0 for i in range(len(relation2id))]
        label[y_id] = 1

        temp = find_index(label, train_ans[tup])
        if temp == -1:
            train_ans[tup].append(label)
            label_tag = len(train_ans[tup]) - 1
            train_sen[tup].append([])
        else:
            label_tag = temp

    sentence = content[3]
    print("sentence: " + sentence)
    en1pos = 0
    en2pos = 0

    # For Chinese
    en1pos = sentence.find(en1)
    if en1pos == -1:
        en1pos = 0
    en2pos = sentence.find(en2)
    if en2pos == -1:
        en2post = 0

    output = []

    # Embeding the position
    for i in range(fixlen):
        word = word2id['BLANK']
        rel_e1 = pos_embed(i - en1pos)
        rel_e2 = pos_embed(i - en2pos)
        output.append([word, rel_e1, rel_e2])

    for i in range(min(fixlen, len(sentence))):
        word = 0
        if sentence[i] not in word2id:
            word = word2id['UNK']
        else:
            word = word2id[sentence[i]]

        output[i][0] = word

    train_sen[tup][label_tag].append(output)
    print(train_sen)
print("=============================================")
print('reading test data ...')

test_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
test_ans = {}  # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)

f = open('./origin_data/testtest.txt', 'r', encoding='utf-8')

while True:
    content = f.readline()
    if content == '':
        break

    content = content.strip().split(' | ')
    en1 = content[0]
    en2 = content[1]
    print("en1: " + en1)
    print("en2: " + en2)
    relation = 0
    if content[2] not in relation2id:
        relation = relation2id['NA']
    else:
        relation = relation2id[content[2]]
    tup = (en1, en2)

    if tup not in test_sen:
        test_sen[tup] = []
        y_id = relation
        label_tag = 0
        label = [0 for i in range(len(relation2id))]
        label[y_id] = 1
        test_ans[tup] = label
    else:
        y_id = relation
        test_ans[tup][y_id] = 1

    sentence = content[3]
    print("sentence: " + sentence)
    en1pos = 0
    en2pos = 0

    # For Chinese
    en1pos = sentence.find(en1)
    if en1pos == -1:
        en1pos = 0
    en2pos = sentence.find(en2)
    if en2pos == -1:
        en2post = 0

    output = []

    for i in range(fixlen):
        word = word2id['BLANK']
        rel_e1 = pos_embed(i - en1pos)
        rel_e2 = pos_embed(i - en2pos)
        output.append([word, rel_e1, rel_e2])

    for i in range(min(fixlen, len(sentence))):
        word = 0
        if sentence[i] not in word2id:
            word = word2id['UNK']
        else:
            word = word2id[sentence[i]]

        output[i][0] = word
    test_sen[tup].append(output)
