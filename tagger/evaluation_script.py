import string

from nltk.corpus import stopwords

nltk_stopwords = set(stopwords.words('english'))


def span_coverage(sh, sr, top_overlap, exact_match=False, position=False, remove_stopwords=True,
                   lemmatization=False):
    """
    calculates coverage of span sh with respect to span sr (partial match)
    sh (spacy nlp): hypothesis span
    sr (spacy nlp): reference span
    position (bool): position matters?
    remove_stopwords (bool):
    """
    # print(sh[0].text, sh[1], sr[0].text, sr[1])
    flag = 0
    if position:
        # print(sh[1],sr[1],len(sr[0].text),len(sh[0].text))
        if sh[1] > sr[1] + len(sr[0]) or sr[1] > sh[1] + len(sh[0]):
            # return 0.0
            flag = 1
    sh_offset = sh[1]
    sr_offset = sr[1]
    # sh_valence = int(sh[3].split("_")[-1].replace("NA","0"))
    # sr_valence = int(sr[3].split("_")[-1].replace("NA","0"))
    sh = sh[0]
    sr = sr[0]

    sh_set = set([token.lower() for token in sh if (
        not (token.lower() in nltk_stopwords and remove_stopwords))])
    sr_set = set([token.lower() for token in sr if (
        not (token.lower() in nltk_stopwords and remove_stopwords))])

    if len(sh_set) == 0 or len(sr_set) == 0:
        return 0.0
    inter = sr_set.intersection(sh_set)
    if flag == 1:
        if len(inter) > 0:
            pass  # do nothing
            # print(sh.text, sh_offset, sr.text, sr_offset)
            # print(inter)
        return 0.0
    overlap_str = " ".join(list(inter))
    if overlap_str not in top_overlap: top_overlap[overlap_str] = []
    # top_overlap[overlap_str]+=[sr_valence, sh_valence]
    # print(inter)
    if exact_match and sr_set == sh_set:
        return 1
    elif exact_match:
        return 0
    # print("sr_set:",len(sr_set)," inter:", len(inter))
    # print(sr_set, sh_set)
    assert (len(inter) / len(sr_set)) <= 1
    return len(inter) / len(sr_set)


def span_set_coverage(SH, SR, exact_match=False, position=False, remove_stopwords=False, lemmatization=False):
    """
    calculates coverage of set of spans SH with respect to a set of spans SR (partial match)
    SH (list): set/list of raw hypothesis spans (from one narrative/segment)
    SR (list): set/list of raw reference spans (from one narrative/segment)
    we need a tuple ('tokens', first token position)
    """
    top_overlap = {}
    coverage = 0
    if len(SH) == 0 and len(SR) == 0: return 1
    for hyp in SH:
        for ref in SR:
            coverage += span_coverage(hyp, ref, top_overlap, exact_match, position, remove_stopwords,
                                            lemmatization)
            """if not exact_match: coverage += span_coverage(hyp, ref, exact_match, position, remove_stopwords, lemmatization)
            elif hyp[0].text == ref[0].text: 
                coverage += 1.0
                #print(hyp[0].text)"""
    return coverage

def from_iob2text(tmp_iob, text, offset=0):
    '''
    :param tmp_iob: iob tags
    :param text: the sentence to split
    :param offset: this is required if you compute the agreement on the whole narrative,
    because we need the real position of each token in the narrative and not in the sentence
    :return:
    '''
    if type(tmp_iob) != list:
        iob = list(tmp_iob)
    else:
        iob = tmp_iob
    if type(text) != list:
        tmp = text.split()
    else:
        tmp = text

    result = []
    com = []
    begin = 0
    err1 = False
    for id_s, tag in enumerate(iob):
        if tag == "b":
            if len(com) != 0:
                result.append((' '.join(com), begin, id_s - 1))
                com = []
                begin = 0
            begin = id_s + offset
            com.append(tmp[id_s])
        elif tag == 'i':
            if len(com) != 0:
                com.append(tmp[id_s])
            else:
                if not err1:
                    print('There is an I without a B')
                    err1 = True
                if len(result) != 0:
                    tmp_com, begin, end = result[-1]
                    tmp_com += ' ' + tmp[id_s]
                    end = id_s - 1
                    result[-1] = (tmp_com, begin, end)
        elif tag == 'o' or id_s == (len(iob) - 1):
            if len(com) != 0:
                result.append((' '.join(com), begin, id_s - 1))
                com = []
                begin = 0
    if len(com) != 0:
        result.append((' '.join(com), begin, len(iob) - 1))

    assert len(result) == iob.count('b')
    return result
def compute_f1_coverage(ref_iob, hyp_iob, sentence, exact_match=False, position=True, remove_stopwords=False, lemmatization=False):
    # clean_sentence = remove_punct(sentence)
    clean_sentence = sentence
    ref = from_iob2text(ref_iob, clean_sentence)
    hyp = from_iob2text(hyp_iob, clean_sentence)
    print(hyp)
    ref_hyp = span_set_coverage(ref, hyp, exact_match=exact_match,
                                position=position, remove_stopwords=remove_stopwords,
                                lemmatization=lemmatization)
    hyp_ref = span_set_coverage(hyp, ref, exact_match=exact_match,
                                position=position, remove_stopwords=remove_stopwords,
                                lemmatization=lemmatization)
    if len(ref) > 0 and len(hyp) > 0:
        precision = ref_hyp / len(hyp)
        recall = hyp_ref / len(ref)

        try:
            f1 = 2 * ((precision * recall) / (precision + recall))
        except:
            print("Warning: zero division happened in compute_f1_coverage")
            f1 = 0

        return f1, precision, recall
    else:
        return 0, 0, 0

def remove_punct(s):
    return s.translate(s.maketrans('', '', string.punctuation))

if __name__ == '__main__':
    print('Case:1', compute_f1_coverage('biioiobii','biioiobii', 'I ran fast at home and I was happy', exact_match=True))
    print('Case:2', compute_f1_coverage('biioiobii', 'biioioooo', 'I ran fast at home and I was happy', exact_match=True))
    print('Case:3', compute_f1_coverage('biioiobii', 'iiioiobii', 'I ran fast at home and I was happy', exact_match=True))
    print('Case:4', compute_f1_coverage('biioiobii', 'iiioioiii', 'I ran fast at home and I was happy', exact_match=True))
    print('Case:5', compute_f1_coverage('biioiobii', 'ooooooooo', 'I ran fast at home and I was happy', exact_match=True))