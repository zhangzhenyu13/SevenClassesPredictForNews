from LanguageModel.WordTokenizer import TokenizerCh,initTokenizer
from DataCleaning.DataLoader import NewsDataset
import pickle

# Default word tokens

PAD_token = 0  # Used for padding short sentences
UNK_token = 1  # after trimed unknown words


#vocabulary model
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", UNK_token: "UNK"}
        self.num_words = 2  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count=3):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", UNK_token: "UNK"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

    def saveVoc(self):
        print("save voc({})".format(len(self.word2count)))
        with open("data/"+self.name+".pkl","wb") as f:
            pickle.dump(self.__dict__,f)

    def loadVoc(self):
        with open("data/"+self.name+".pkl","rb") as f:
            self.__dict__=pickle.load(f)
        print("load voc({})".format(len(self.word2count)))

def buildVoc(tokenizer_ch,nwdata):
    X, Y = nwdata.fetchAll(cols=[1, 3])
    texts = []
    for x in X.iloc[:,0]:
        texts.append(x)
    for x in X.iloc[:,1]:
        texts.append(x)

    cleaned = tokenizer_ch.cleanDocs(texts)
    # print("cleaned\n",cleaned)
    voc = Voc("news_voc")

    [voc.addSentence(" ".join(tks)) for tks in cleaned]

    voc.saveVoc()
    return voc

def testFunc():
    tokenizer_ch=initTokenizer()
    nwdata = NewsDataset("data/trainex.xls")

    voc=buildVoc(tokenizer_ch,nwdata)

    print(voc.num_words)
    voc.trim()
    print(voc.num_words)

if __name__ == '__main__':
    testFunc()
