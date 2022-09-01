import json 
import stanza 

def processFiles(dataset, mode):
    dict = {}
    sentence_packs = json.load(open(dataset + '/' + mode +'.json'))
    count = 0

    for sentence_pack in sentence_packs:
        sentence =  sentence_pack['sentence']
        triples =  sentence_pack['triples']
        pos_tags =  sentence_pack['pos_tags']
        heads =  sentence_pack['heads']
        children =  sentence_pack['children']
        relations =  sentence_pack['relations']

        words = sentence.split(" ")
        for i in range(len(words)):
            if "DT" in pos_tags[i] and "NN" in pos_tags[heads[i]] and relations[i] == "det":
                count += 1

        # for triple in triples:
            # aspect, opinion, sentiment = triple[0], triple[1], triple[2]
            
            # if aspect[0] == aspect[1] and opinion[0] == opinion[1]:
                """
                if heads[aspect[0]] == opinion[0]:
                    if "DT" in pos_tags[aspect[0]] and "NN" in pos_tags[opinion[0]] and relations[aspect[0]] == "det":
                        count += 1
                elif heads[opinion[0]] == aspect[0]:
                    if "DT" in pos_tags[opinion[0]] and "NN" in pos_tags[aspect[0]] and relations[opinion[0]] == "det":
                        count += 1
                """

                """
                if heads[aspect[0]] == opinion[0]:
                    index = pos_tags[aspect[0]] + "<-" + relations[aspect[0]] + "-"+ pos_tags[opinion[0]]
                    if index not in dict:
                        dict[index] = 1
                    else:
                        dict[index] += 1
                elif heads[opinion[0]] == aspect[0]:
                    index = pos_tags[opinion[0]] + "<-" + relations[opinion[0]] + "-"+ pos_tags[aspect[0]]
                    if index not in dict:
                        dict[index] = 1
                    else:
                        dict[index] += 1
                """
    #dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    return count



    # file = open(dataset + mode+".json",'w')
    # file.write(json.dumps(sentences))
    # file.close()

if __name__ == "__main__":
    # data preprocessing
    count1 = processFiles("res14", "train")
    print(count1)

    count1 = processFiles("res15", "train")
    print(count1)

    count1 = processFiles("res16", "train")
    print(count1)

    count1 = processFiles("lap14", "train")
    print(count1)

    # file = open("dict_pos.json",'w')
    # file.write(json.dumps(dict_pos))
    # file.close()
