
import argparse
import pprint

from data import Read



class Evaluate:
    def __init__(self, args) -> None:

        # print all args
        pprint.pprint(vars(args))    

        self.pred = Read(file_name=args.pred_file,
                         language=args.language,
                         mode=args.mode,
                         eval_bool=True)

        self.gold = Read(file_name=args.gold_file,
                         language=args.language,
                         mode=args.mode)

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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="dev",
        help="evaluate on dev set vs. test set",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help="language of data (english vs. german)",
    )

    parser.add_argument(
        "--pred_file",
        type=str,
        help="file with predictions (filename only)",
    )

    parser.add_argument(
        "--gold_file",
        type=str,
        default="wsj_dev.conll06.gold",
        help="file with gold labels",
    )


    args = parser.parse_args()
    evaluation = Evaluate(args)

    print("\nUAS:", evaluation.UAS())
    #print("LAS:", evaluation.LAS())