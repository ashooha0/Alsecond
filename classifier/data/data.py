import json 
import stanza 

def processFiles(dataset, mode):
    count, sentences = 0, []
    nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma, depparse', tokenize_pretokenized=True, tokenize_no_ssplit = True)
    
    file = open(dataset + '/' + mode +'_triplets.txt')
    line = file.readline() 
    while line:
        index = line.find("####")
        sentence = line[:index]
        triples = line[index+4:]

        new_triples = []
        triples = triples[1:-2]
        triples = triples.split(")")[:-1]
        for triple in triples:
            sentiment_index = triple.find("'")
            sentiment = triple[sentiment_index+1:-1]

            triple = triple[:sentiment_index-2]
            aspect_start_index = triple.find("[")
            aspect_end_index = triple.find("]")
            aspect = triple[aspect_start_index+1:aspect_end_index]
            aspect = aspect.split(",")
            aspect = [int(a) for a in aspect]

            triple = triple[aspect_end_index+3:]
            opinion_start_index = triple.find("[")
            opinion_end_index = triple.find("]")
            opinion = triple[opinion_start_index+1:opinion_end_index]
            opinion = opinion.split(",")
            opinion = [int(o) for o in opinion]

            triple = [[aspect[0], aspect[-1]], [opinion[0], opinion[-1]], sentiment]
            new_triples.append(triple)
        # print(new_triples)

        pos_tags, relations, heads = [], [], []
        doc = nlp(sentence)
        for sent in doc.sentences:
            for word in sent.words:
                pos_tags.append(word.xpos)
                relations.append(word.deprel)
                heads.append(word.head-1)
                
        words = sentence.split(" ")
        children = [[] for i in range(len(words))]
        for i in range(len(words)):
            if heads[i] != -1:
                head = heads[i]
                children[head].append(i)
        
        new = {                                         # Example:
            'id': count,                                # 0
            'sentence': sentence,                       # But the staff was so horrible to us . 
            'triples': new_triples,                     # [[[2, 2], [5, 5], 'NEG']]
            'pos_tags': pos_tags,                       # ['CC', 'DT', 'NN', 'VBD', 'RB', 'JJ', 'IN', 'PRP', '.']
            'heads': heads,                             # [5, 2, 5, 5, 5, -1, 7, 5, 5]
            'children': children,                       # [[], [], [1], [], [], [0, 2, 3, 4, 7, 8], [], [6], []]
            'relations': relations,                     # ['cc', 'det', 'nsubj', 'cop', 'advmod', 'root', 'case', 'obl', 'punct']
        }
        sentences.append(new)
        count += 1

        line = file.readline() 

    file = open(dataset + mode+".json",'w')
    file.write(json.dumps(sentences))
    file.close()

if __name__ == "__main__":
    # data preprocessing
    processFiles("14res", "train")
    processFiles("14res", "dev")
    processFiles("14res", "test")

    processFiles("15res", "train")
    processFiles("15res", "dev")
    processFiles("15res", "test")

    processFiles("16res", "train")
    processFiles("16res", "dev")
    processFiles("16res", "test")

    processFiles("14lap", "train")
    processFiles("14lap", "dev")
    processFiles("14lap", "test")