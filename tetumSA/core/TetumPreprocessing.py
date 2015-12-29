import nltk
import re

#####
##
## Tokenizers
##
#####

def tokenize_tokens(text):
    """Return list of tokens for text string input"""
    return nltk.tokenize.word_tokenize(text)

def tokenize_sentences(text):
    """Returns a list of sentences (as strings) from the input text"""
    return nltk.tokenize.sent_tokenize(text)

#####
##
## Token processing
##
## The following functions operate on lists of tokens.
##
#####

def fold_case(tokens):
    """Fold tokens to lower case"""
    return [tok.lower() for tok in tokens]


def filter_tokens(tokens, min_size=0, special_chars=False):
    """Filter list of tokens.

    Any token with length below *min_size* is removed. If *special_chars*
    is ``True``, all tokens containing special chars will also be removed.
    """
    if min_size>0:
        tokens = [t for t in tokens if len(t) >= min_size]
    if special_chars:
        tokens = [t for t in tokens if re.search('[^a-zA-Z-]',t)==None]
    return tokens

#####
##
## Processing methods
##
#####

def preprocess_token(token, fold=True, specials=True, min_size=3):
    """Perform preprocessing on a single token."""
    if fold: token = fold_case([token])[0]
    filt = filter_tokens([token], min_size, specials)
    if len(filt)==0: return None
    else: return filt[0]

def preprocess_dataset(dataset, fold=True, specials=True, min_size=3):
    """Perform preprocessing steps on the input text."""
    featuresets = []
    for text in dataset:
        processedText = preprocess_text(text,fold,specials,min_size)
        featuresets.append(processedText)
    return featuresets

def preprocess_text(text,fold=True, specials=True, min_size=3):
    """Perform preprocessing steps on the input text."""
    processedText=[]
    if text == None:
        return processedText
    sentences = tokenize_sentences(text)
    for sentence in sentences:
        processedSentence = preprocess_sentence(sentence,fold,specials,min_size)
        processedText+=processedSentence
    return processedText

def preprocess_sentence(text, fold=True, specials=True, min_size=3):
    """Perform preprocessing steps on the input text."""
    ts = tokenize_tokens(text)
    ts = filter_tokens(ts, min_size=min_size,special_chars=specials)
    if fold:
        ts = fold_case(ts)
    return ts


#####
##
## Tests
##
## Simple tests of functions in this module.
##
#####

def test_tokenize_tokens():
    text = "This is sentence one."
    tok = tokenize_tokens(text)
    assert(tok==['This', 'is', 'sentence', 'one', '.'])

def test_tokenize_sentences():
    text = "This is sentence one. And this, my friend, is sentence two. And this here -- with an paranthetical dash sentence shot in -- is the last sentence."
    ss = tokenize_sentences(text)
    assert(ss[0]=="This is sentence one.")
    assert(ss[1]=="And this, my friend, is sentence two.")
    assert(ss[2]=="And this here -- with an paranthetical dash sentence shot in -- is the last sentence.")


def test_fold_case():
    pre = ['Hi!', 'MiXeD', 'caSE']
    post = ['hi!', 'mixed', 'case']
    assert(fold_case(pre)==post)
    
def test_filter_tokens():
    pre = ['ab', 'abc', '1abc', '12']
    assert(filter_tokens(pre, min_size=3)==['abc', '1abc'])
    assert(filter_tokens(pre, special_chars=True)==['ab', 'abc'])
    assert(filter_tokens(pre, min_size=3, special_chars=True)==['abc'])
def test_preprocess_dataset():
    dataset = ["Laran kontenti tebes atu hasoru malu ho ita Obrigo burodo. Sorte diak ba loron ohin."]
    results = preprocess_dataset(dataset,True,False,2)
    print results
def test_preprocess_sentence():
    text = "Laran kontenti tebes atu hasoru malu ho ita Obrigo burodo. Sorte diak ba loron ohin."
    processed = ['laran', 'kontenti', 'tebes', 'atu', 'hasoru', 'malu', 'ho', 'ita', 'obrigo', 'burodo', 'sorte', 'diak', 'ba', 'loron', 'ohin']
    assert(preprocess_sentence(text)==processed)

def run_tests():
    test_tokenize_sentences()
    test_tokenize_tokens()
    test_fold_case()
    test_filter_tokens()
    test_preprocess_sentence()
    print "ok"

# -----

if __name__ == "__main__":
    #run_tests()
    test_preprocess_dataset()
