from functools import total_ordering, reduce
import csv
import re


@total_ordering
class Posting:

    def __init__(self, docID):
        self._docID = docID

    def get_from_corpus(self, corpus):
        return corpus[self._docID]

    def __eq__(self, other):
        """Perform the comparison between this posting and another one.
        Since the ordering of the postings is only given by their docID,
        they are equal when their docIDs are equal.
        """
        return self._docID == other._docID

    def __gt__(self, other):
        """As in the case of __eq__, the ordering of postings is given
        by the ordering of their docIDs
        """
        return self._docID > other._docID

    def __repr__(self):
        return str(self._docID)


class PostingList:

    def __init__(self):
        self._postings = []

    @classmethod
    def from_docID(cls, docID):
        """ A posting list can be constructed starting from
        a single docID.
        """
        plist = cls()
        plist._postings = [Posting(docID)]
        return plist

    @classmethod
    def from_posting_list(cls, postingList):
        """ A posting list can also be constructed by using another
        """
        plist = cls()
        plist._postings = postingList
        return plist

    def merge(self, other):
        """Merge the other posting list to this one in a desctructive
        way, i.e., modifying the current posting list. This method assume
        that all the docIDs of the second list are higher than the ones
        in this list. It assumes the two posting lists to be ordered
        and non-empty. Under those assumptions duplicate docIDs are
        discarded
        """
        i = 0
        last = self._postings[-1]
        while (i < len(other._postings) and last == other._postings[i]):
            i += 1
        self._postings += other._postings[i:]

    def intersection(self, other):
        """Returns a new posting list resulting from the intersection
        of this one and the one passed as argument.
        """
        intersection = []
        i = 0
        j = 0
        while (i < len(self._postings) and j < len(other._postings)):
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
        """Returns a new posting list resulting from the union of this
        one and the one passed as argument.
        """
        union = []
        i = 0
        j = 0
        while (i < len(self._postings) and j < len(other._postings)):
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
        for k in range(j, len(other._postings)):
            union.append(other._postings[k])
        return PostingList.from_posting_list(union)

    def get_from_corpus(self, corpus):
        return list(map(lambda x: x.get_from_corpus(corpus), self._postings))

    def __repr__(self):
        return ", ".join(map(str, self._postings))


def normalize(text):
    """ A simple funzion to normalize a text.
    It removes everything that is not a word, a space or an hyphen
    and downcases all the text.
    """
    no_punctuation = re.sub(r'[^\w^\s^-]', '', text)
    downcase = no_punctuation.lower()
    return downcase


def tokenize(movie):
    """ From a movie description returns a posting list of all
    tokens present in the description.
    """
    text = normalize(movie.description)
    return list(text.split())


class ImpossibleMergeError(Exception):
    pass


@total_ordering
class Term:

    def __init__(self, term, docID):
        self.term = term
        self.posting_list = PostingList.from_docID(docID)

    def merge(self, other):
        """Merge (destructively) this term and the corresponding posting list
        with another equal term and its corrsponding posting list.
        """
        if (self.term == other.term):
            self.posting_list.merge(other.posting_list)
        else:
            raise ImpossibleMergeError()

    def __eq__(self, other):
        return self.term == other.term

    def __gt__(self, other):
        return self.term > other.term

    def __repr__(self):
        return self.term + ": " + repr(self.posting_list)


class InvertedIndex:

    def __init__(self):
        self._dictionary = []  # A collection of terms

    @classmethod
    def from_corpus(cls, corpus):
        # Here we "cheat" by using python dictionaries
        intermediate_dict = {}
        for docID, document in enumerate(corpus):
            tokens = tokenize(document)
            for token in tokens:
                term = Term(token, docID)
                try:
                    intermediate_dict[token].merge(term)
                except KeyError:
                    intermediate_dict[token] = term
            # To observe the progress of our indexing.
            if (docID % 1000 == 0):
                print("ID: " + str(docID))
        idx = cls()
        idx._dictionary = sorted(intermediate_dict.values())
        return idx

    def __getitem__(self, key):
        for term in self._dictionary:
            if term.term == key:
                return term.posting_list
        raise KeyError

    def __repr__(self):
        return "A dictionary with " + str(len(self._dictionary)) + " terms"


class MovieDescription:

    def __init__(self, title, description):
        self.title = title
        self.description = description

    def __repr__(self):
        return self.title  # + "\n" + self.description + "\n"


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
                # We ignore the descriptions for which we cannot find a title
                pass
        return corpus


class IRsystem:

    def __init__(self, corpus, index):
        self._corpus = corpus
        self._index = index

    @classmethod
    def from_corpus(cls, corpus):
        index = InvertedIndex.from_corpus(corpus)
        return cls(corpus, index)

    def answer_query(self, words):
        norm_words = map(normalize, words)
        postings = []
        for w in norm_words:
            try:
                res = self._index[w]
            except KeyError:
                res = []
            postings.append(res)
        plist = reduce(lambda x, y: x.intersection(y), postings)
        return plist.get_from_corpus(self._corpus)

    def answer_query_sc(self, words):
        norm_words = map(normalize, words)
        postings = []
        for w in norm_words:
            try:
                res = self._index[w]
            except KeyError:
                dictionary = [t.term for t in self._index._dictionary]
                sub = find_nearest(w, dictionary, keep_first=True)
                print("{} not found. Did you mean {}?".format(w, sub))
                res = self._index[sub]
            postings.append(res)
        plist = reduce(lambda x, y: x.intersection(y), postings)
        return plist.get_from_corpus(self._corpus)


def edit_distance(u, v):
    nrows = len(u) + 1
    ncols = len(v) + 1
    M = [[0] * ncols for i in range(0, nrows)]
    for i in range(0, nrows):
        M[i][0] = i
    for j in range(0, ncols):
        M[0][j] = j
    for i in range(1, nrows):
        for j in range(1, ncols):
            candidates = [M[i-1][j] + 1, M[i][j-1] + 1]
            if (u[i-1] == v[j-1]):
                candidates.append(M[i-1][j-1])
            else:
                candidates.append(M[i-1][j-1] + 1)
            M[i][j] = min(candidates)
            # Remove the comments to print the distance matrix
            # print(M[i][j], end="\t")
        # print()
    return M[-1][-1]  # Bottom right element of M


def find_nearest(word, dictionary, keep_first=False):
    if keep_first:
        # If keep_first is true then we only search across the words
        # in the dictionary starting with the same letter
        dictionary = list(filter(lambda w: w[0] == word[0], dictionary))
    # Remove comment to see the reduction in the size of the dictionary
    # when keeping fixed the first letter
    # print(len(dictionary))
    # Apply f(x) = edit_distance(word, x) to all words in the dictionary
    distances = map(lambda x: edit_distance(word, x), dictionary)
    # Produce all the pairs (distance, term) usng zip and find one with
    # the minimal distance.
    return min(zip(distances, dictionary))[1]


def query(ir, text):
    words = text.split()
    answer = ir.answer_query_sc(words)
    for movie in answer:
        print(movie)
