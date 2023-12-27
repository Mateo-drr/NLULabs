from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
from nltk.corpus import wordnet_ic
from nltk.metrics.scores import precision, recall, f_measure, accuracy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import senseval

semcor_ic = wordnet_ic.ic('ic-semcor.dat')

mapping = {
    'interest_1': 'interest.n.01',
    'interest_2': 'interest.n.03',
    'interest_3': 'pastime.n.01',
    'interest_4': 'sake.n.01',
    'interest_5': 'interest.n.05',
    'interest_6': 'interest.n.04',
}

def preprocess(text):
    mapping = {"NOUN": wordnet.NOUN, "VERB": wordnet.VERB, "ADJ": wordnet.ADJ, "ADV": wordnet.ADV}
    sw_list = stopwords.words('english')
    
    lem = WordNetLemmatizer()
    
    # tokenize, if input is text
    tokens = nltk.word_tokenize(text) if type(text) is str else text
    # pos-tag
    tagged = nltk.pos_tag(tokens, tagset="universal")
    # lowercase
    tagged = [(w.lower(), p) for w, p in tagged]
    # optional: remove all words that are not NOUN, VERB, ADJ, or ADV (i.e. no sense in WordNet)
    tagged = [(w, p) for w, p in tagged if p in mapping]
    # re-map tags to WordNet (return orignal if not in-mapping, if above is not used)
    tagged = [(w, mapping.get(p, p)) for w, p in tagged]
    # remove stopwords
    tagged = [(w, p) for w, p in tagged if w not in sw_list]
    # lemmatize
    tagged = [(w, lem.lemmatize(w, pos=p), p) for w, p in tagged]
    # unique the list
    tagged = list(set(tagged))
    return tagged

def get_sense_definitions(context):
    # input is text or list of strings
    lemma_tags = preprocess(context)
    # let's get senses for each
    senses = [(w, wordnet.synsets(l, p)) for w, l, p in lemma_tags]
    
    # let's get their definitions
    definitions = []
    for raw_word, sense_list in senses:
        if len(sense_list) > 0:
            # let's tokenize, lowercase & remove stop words 
            def_list = []
            for s in sense_list:
                defn = s.definition()
                # let's use the same preprocessing
                tags = preprocess(defn)
                toks = [l for w, l, p in tags]
                def_list.append((s, toks))
            definitions.append((raw_word, def_list))
    return definitions

def get_top_sense(words, sense_list):
    # get top sense from the list of sense-definition tuples
    # assumes that words and definitions are preprocessed identically
    val, sense = max((len(set(words).intersection(set(defn))), ss) for ss, defn in sense_list)
    return val, sense

def original_lesk(context_sentence, ambiguous_word, pos=None, synsets=None, majority=False):

    context_senses = get_sense_definitions(set(context_sentence)-set([ambiguous_word]))
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    scores = []
    # print(synsets)
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense(sense[1], synsets))
            
    if len(scores) == 0:
        return synsets[0][0]
    
    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    return best_sense

def get_top_sense_sim(context_sense, sense_list, similarity):
    # get top sense from the list of sense-definition tuples
    # assumes that words and definitions are preprocessed identically
    scores = []
    for sense in sense_list:
        ss = sense[0]
        if similarity == "path":
            try:
                scores.append((context_sense.path_similarity(ss), ss))
            except:
                scores.append((0, ss))    
        elif similarity == "lch":
            try:
                scores.append((context_sense.lch_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "wup":
            try:
                scores.append((context_sense.wup_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "resnik":
            try:
                scores.append((context_sense.res_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lin":
            try:
                scores.append((context_sense.lin_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "jiang":
            try:
                scores.append((context_sense.jcn_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        else:
            print("Similarity metric not found")
            return None
    val, sense = max(scores)
    return val, sense

def lesk_similarity(context_sentence, ambiguous_word, similarity="resnik", pos=None, 
                    synsets=None, majority=True):
    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))
    
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    
    scores = []
    
    # Here you may have some room for improvement
    # For instance instead of using all the definitions from the context
    # you pick the most common one of each word (i.e. the first)
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense_sim(sense[0], synsets, similarity))
            
    if len(scores) == 0:
        return synsets[0][0]
    
    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    
    return best_sense

def leskEval(data,option):
    refs = {k: set() for k in mapping.values()}
    hyps = {k: set() for k in mapping.values()}
    refs_list = []
    hyps_list = []
    
    # since WordNet defines more senses, let's restrict predictions
    
    synsets = []
    for ss in wordnet.synsets('interest', pos='n'):
        if ss.name() in mapping.values():
            defn = ss.definition()
            tags = preprocess(defn)
            toks = [l for w, l, p in tags]
            synsets.append((ss,toks))
    
    for i, inst in enumerate(data):
        txt = [t[0] for t in inst.context]
        raw_ref = inst.senses[0] # let's get first sense
        if option==1:
            hyp = original_lesk(txt, txt[inst.position], synsets=synsets, majority=True).name()
        else:
            hyp = lesk_similarity(txt, txt[inst.position], synsets=synsets, majority=True).name()
        ref = mapping.get(raw_ref)
        
        # for precision, recall, f-measure        
        refs[ref].add(i)
        hyps[hyp].add(i)
        
        # for accuracy
        refs_list.append(ref)
        hyps_list.append(hyp)
    
        ac=accuracy(refs_list, hyps_list)
    print("Acc:", round(ac, 3))
    
    
    for cls in hyps.keys():
        met = []
        met.append(precision(refs[cls], hyps[cls]))
        met.append(recall(refs[cls], hyps[cls]))
        met.append(f_measure(refs[cls], hyps[cls], alpha=1))
        for i in range(len(met)):
            if met[i] is None:
                met[i]=0
        
        print("{:15s}: p={:.3f}; r={:.3f}; f={:.3f}; s={}".format(cls, met[0], met[1], met[2], len(refs[cls])))
        
    return ac
    
def dataproc():
    #Process data to get a list of sentences (data) and what meaning the word of interest has (lbls)
    data = [" ".join([t[0] for t in inst.context]) for inst in senseval.instances('interest.pos')]
    lbls = [inst.senses[0] for inst in senseval.instances('interest.pos')]
    #data and label encoder
    vectorizer = CountVectorizer()
    lblencoder = LabelEncoder()
    #K fold 
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)
    #Get the encoded data
    vectors = vectorizer.fit_transform(data)
    # encoding labels for multi-calss
    lblencoder.fit(lbls)
    labels = lblencoder.transform(lbls)
    
    return data,lbls,vectors,labels, stratified_split

def collocational_features_n(inst, n, pos):
    p = inst.position
    features = {}
    # Iterate from -n to +n
    for i in range(-n, n + 1):
        #Solution for words close to the beggining and end of sentences
        if p + i < 0 or p + i >= len(inst.context): # If the position is out of bounds then NULL
            features[f'w{i}_word'] = 'NULL'
            if pos:
                features[f'w{i}_p'] = 'NULL'
        else: #Just add the actual pos and context word
            features[f'w{i}_word'] = inst.context[p + i][0]
            if pos:
                features[f'w{i}_p'] = inst.context[p + i][1]

    return features

def getJointData(instances, n):
    data_col = []
    for inst in instances:
        features = collocational_features_n(inst, n, True)
        data_col.append(features)
    
    #Vectorizer for the context features dict
    dvectorizer = DictVectorizer(sparse=False)
    dvectors = dvectorizer.fit_transform(data_col)
    return dvectors

    