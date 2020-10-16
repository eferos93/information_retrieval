# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

from functools import total_ordering, reduce
import re
import csv


class MovieDescription:

    def __init__(self, title, description):
        self.title = title
        self.description = description

    def __repr__(self):
        return self.title


    def read_movie_descriptions():
        filename = 'plot_summaries.txt'
        movie_names_file = 'movie.metadata.tsv'
        with open(movie_names_file, 'r') as csv_file:
            movie_names = csv.reader(csv_file, delimiter='\t')
        names_table = {}
        for name in movie_names:
            names_table[name[0]] = name[2]
        with open(filename, 'r') as csv_file:
            descriptions = csv.reader(csv_file, delimiter='\t')
        corpus = []
        for desc in descriptions:
            try:
                movie = MovieDescription(names_table[desc[0]], desc[1])
                corpus.append(movie)
            except KeyError:
                pass
        return corpus

@total_ordering
class Posting:
    def __init__(self, documentId):
        self.documentId = documentId

    def get_from_corpus(self, corpus):
        return corpus[self.documentId]

    def __eq__(self, other):
        return self.documentId == other.documentId

    def __gt__(self, other):
        return self.documentId > other.documentId

    def __repr__(self):
        return str(self.documentId)

class PostingList:
    def __init__(self):
        self._postings = []

    @classmethod
    def from_docId(cls, docId):
        plist = cls()
        plist._postings = [(Posting(docId))]
        return plist

    @classmethod
    def from_posting_list(cls, postingList):
        plist = cls()
        plist._postings = postingList
        return  plist

    def merge(self, other):
        i = 0
        last = self._postings[-1]
        while (i < len(other._postings) and last == other._postings[i]):
            i += 1
        self._postings += other._postings[i:]

    def intersection(self, other):
        intersection = []
        i = j = 0
        while ( i < len(self._postings) and j < len(other._postings)):
            if (self._postings[i] == other._postings[j]):
                intersection.append(self._postings[i])
                i += 1
                j += 1
            elif (self._postings[i] < other._postings[j]):
                i += 1
            else:
                j += 1
        return PostingList.from_posting_list(intersection)

    def union(self, other):
        union = []
        i = j = 0
        while(i < len(self._postings) and j < len(other._postings)):
            if (self._postings[i] == other._postings[j]):
                union.append(self._postings[i])
                i += 1
                j += 1
            elif (self._postings[i] < other._postings[j]):
                union.append(self._postings[i])
                i += 1
            else:
                union.append(other._postings[j])
                j += 1

        for k in range(i, len(self._postings)):
            union.append(self._postings[k])
        for k in range(i, len(other._postings)):
            union.append(other._postings[k])

        return PostingList.from_posting_list(union)

    def get_from_corpus(self, corpus):
        return list(map(lambda x: x.get_from_corpus(corpus), self._postings))

    def __repr__(self):
        return ", ".join(map(str, self._postings))

#TERMS
class ImpossibleMergeException(Exception):
    pass

@total_ordering
class Term:
    def __init__(self, term, docId):
        self.term = term
        self.posting_list = PostingList.from_docId(docId)

    def merge(self, other):
        if(self.term == other.term):
            self.posting_list.merge(other.posting_list)
        else:
            raise ImpossibleMergeException

    def __eq__(self, other):
        return self.term == other.term

    def __gt__(self, other):
        return self.term > other.term

    def __repr__(self):
        return self.term + ": " + repr(self.posting_list)


def normalise(text):
    no_punctuation = re.sub(r'[^\w^\s^-]', '', text) #remove everything that is not a word, not a space and not a -
    downcase = no_punctuation.lower()
    return downcase

def tokenise(movie):
    text = normalise(movie.description)
    return list(text.split())

print(tokenise(MovieDescription("title", "This is a movie Description")))


class InvertedIndex:
    def __init__(self):
        self._dictionary = []

    @classmethod
    def from_corpus(cls, corpus):
        intermidiate_dict = {}
        for docId, document in enumerate(corpus):
            tokens = tokenise(document)
            for token in tokens:
                term = Term(token, docId)
                try:
                    intermidiate_dict[token].merge(term)
                except KeyError:
                    intermidiate_dict[token] = term
            if (docId%1000 == 0):
                print("ID: " + str(docId))
        idx = cls()
        idx._dictionary = sorted(intermidiate_dict.values())
        return idx

    def __getitem__(self, item):
        for term in self._dictionary:
            if term.term == item:
                return term.posting_list
        raise KeyError

    def __repr__(self):
        return "A dictionary with " + str(len(self._dictionary)) + " terms"


class IRSystem:
    def __init__(self, corpus, index):
        self._corpus = corpus
        self._index = index

    @classmethod
    def from_corpus(cls, corpus):
        index = InvertedIndex.from_corpus(corpus)
        return  cls(corpus, index)

    def answer_query(self, words):
        norm_words = map(normalise, words)
        postings = map(lambda w: self._index[w], norm_words)
        plist = reduce(lambda x, y: x.intersection(y), postings)
        return plist.get_from_corpus(self._corpus)

def query(ir, text):
    words = text.split()
    answer = ir.answer_query(words)
    for movie in answer:
        print(movie)

ir = IRSystem



