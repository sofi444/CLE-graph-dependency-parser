
import sys

from data import Read



#n_args = len(sys.argv)
#pred_file = sys.argv[1]
#gold_file = sys.argv[2]
#language = sys.argv[3]
#mode = sys.argv[4]



class Evaluate:
    def __init__(self, pred_file, gold_file, language, mode) -> None:

        self.pred = Read(pred_file, language, mode)
        self.gold = Read(gold_file, language, mode)

        assert len(self.pred.all_sentences) == len(self.gold.all_sentences), "Length mismatch"


    def UAS(self):
        self.correct_heads = 0
        self.total_tokens = 0

        for sent_idx in range(len(self.pred.all_sentences)):
            pred_heads = self.pred.all_sentences[sent_idx].head #list
            gold_heads = self.gold.all_sentences[sent_idx].head #list

            for token_idx in range(1, len(pred_heads)): # Skip ROOT
                self.total_tokens += 1
                if pred_heads[token_idx] == gold_heads[token_idx]:
                    self.correct_heads += 1
        
        return self.correct_heads / self.total_tokens


    def LAS(self):
        self.correct_heads_and_labels = 0
        self.total_tokens = 0

        for sent_idx in range(len(self.pred.all_sentences)):
            pred_heads = self.pred.all_sentences[sent_idx].head #list
            gold_heads = self.gold.all_sentences[sent_idx].head #list
            pred_rels = self.pred.all_sentences[sent_idx].rel #list
            gold_rels = self.gold.all_sentences[sent_idx].rel #list

            for token_idx in range(1, len(pred_heads)): # Skip ROOT
                self.total_tokens += 1
                if pred_heads[token_idx] == gold_heads[token_idx]:
                    if pred_rels[token_idx] == gold_rels[token_idx]:
                        self.correct_heads_and_labels += 1
        
        return self.correct_heads_and_labels / self.total_tokens




if __name__ == "__main__":

    pred_file = "wsj_dev.conll06.pred"
    gold_file = "wsj_dev.conll06.gold"
    language = "english"
    mode = "dev"


    evaluation = Evaluate(pred_file, gold_file, language, mode)
    print("UAS:", evaluation.UAS())
    print("LAS:", evaluation.LAS())