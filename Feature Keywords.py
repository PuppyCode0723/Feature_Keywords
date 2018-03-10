import re

# 處理繁體中文
import jieba2 as jieba
import jieba2.analyse
import sqlite3
import pickle
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC


# 回傳停用詞
def stopwordslist(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    return stopwords


# 移除停用詞
def removestopwords(content, stopwords):
    non_stopwords = []
    for i in content:
        if i not in stopwords:
            non_stopwords.append(i)

    return non_stopwords


stopwords = stopwordslist('./stopwords.txt')

conn = sqlite3.connect('./dataSet/PTT Data Set.db')
cursor = conn.cursor()

pushes = cursor.execute('SELECT Title, Push_tag FROM PTT_Gossiping')

# Title, Push_tag

scores = []
temp = None
temp_score = 0

count = 0

for push in pushes:
    if temp is None:
        temp = push[0]

    if push[0] in temp:
        if push[1] == '推':
            temp_score += 1
        elif push[1] == '噓':
            temp_score -= 1
    elif push[0] not in temp:
        scores.append(temp_score)
        temp = push[0]
        if count % 150 == 0:
            print('temp_score: ', temp_score)
        temp_score = 0

    count += 1

print(len(scores))
rlt = []

corpus = cursor.execute('SELECT Title, Content FROM PTT_Gossiping ')
temp_title = None
words = defaultdict(int)

for content in corpus:
    if temp_title is None:
        temp_title = content[0]

    if content[0] in temp_title:
        if content[1] is None:
            continue
        else:
            old_sentence = str(content[1]).replace("\\n", '')
            # 只保留繁體字
            new_sentence = re.sub(r'[^\u4e00-\u9fa5]', '', old_sentence)
            stopword_sentence = []
            for w in jieba2.cut(new_sentence):
                stopword_sentence.append(w)
            non_stopword_sentence = removestopwords(stopword_sentence, stopwords)
            for w in non_stopword_sentence:
                words[w] += 1
    elif content[0] not in temp_title:
        rlt.append(words)
        words = defaultdict(int)
        temp_title = content[0]

print(len(rlt))

dvec = DictVectorizer()
tfidf = TfidfTransformer()
X = tfidf.fit_transform(dvec.fit_transform(rlt))

svc = LinearSVC()
svc.fit(X, scores)
with open('./SVM_LABEL.pickle', 'wb') as f:
    pickle.dump(dvec.get_feature_names(), f)
with open('./SVM_coef.pickle', 'wb') as f:
    t = []
    for i in svc.coef_[0]:
        t.append(i)
    pickle.dump(t, f)

print(dvec.get_feature_names()[:10])
print(svc.coef_[0][:10])
