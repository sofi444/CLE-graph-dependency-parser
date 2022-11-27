import sys
import os



n_args = len(sys.argv)
#file_name = sys.argv[1]
#language = sys.argv[2]
#mode = sys.argv[3]
file_name = "wsj_train.first-1k.conll06"
language = "english"
mode = "train"



class Sentence:
    def __init__(self, sentence_raw:list) -> None:

        '''Add Root'''
        self.id = ["0"]
        self.form = ["ROOT"]
        self.lemma = ["ROOT"]
        self.pos = ["_"]
        self.xpos = ["_"]
        self.morph = ["_"]
        self.head = ["_"]
        self.rel = ["_"]
        self.empty1 = ["_"] #?
        self.empty2 = ["_"] #?

        '''Fill Sentence object'''
        for token in sentence_raw:
            tags = token.split("\t") #tags:list
            self.id.append(tags[0])
            self.form.append(tags[1])
            self.lemma.append(tags[2])
            self.pos.append(tags[3])
            self.xpos.append(tags[4])
            self.morph.append(tags[5])
            self.head.append(tags[6])
            self.rel.append(tags[7])
            self.empty1.append(tags[8])
            self.empty2.append(tags[9])



class Read:
    def __init__(self, file_name:str, language:str, mode:str) -> None:
        
        '''Set Path'''
        self.file_name = file_name
        self.file_path = os.path.join(
            f"data/{language}/{mode}/",
            self.file_name
        )

        '''Parse file'''
        self.all_sentences = []

        with open(self.file_path, "r") as f:
            data = f.readlines()

            sentence = []
            for line in data:
                if line != "\n": # Newlines separate sentences
                    sentence.append(line.strip("\n")) # line:str
                else:
                    sentence_obj = Sentence(sentence)
                    sentence = []
                    self.all_sentences.append(sentence_obj)
                


class Write:
    def __init__(self, outfile_name:str, out_content:list) -> None:
        self.outfile_name = outfile_name
        self.out_path = os.path.join(f"data/", self.outfile_name)
        self.out_content = out_content
        
        with open(self.out_path, "w") as f:

            for sent in self.out_content:
                for i in range(1, len(sent.id)):
                    f.write(
                        sent.id[i] + "\t" +
                        sent.form[i] + "\t" +
                        sent.lemma[i] + "\t" +
                        sent.pos[i] + "\t" +
                        sent.xpos[i] + "\t" +
                        sent.morph[i] + "\t" +
                        sent.head[i] + "\t" +
                        sent.rel[i] + "\t" +
                        sent.empty1[i] + "\t" +
                        sent.empty2[i]
                    )
                f.write("\n")



        



reader = Read(file_name, language, mode)
Write("sanity_check.conll06", reader.all_sentences)