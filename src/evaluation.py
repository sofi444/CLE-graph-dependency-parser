
import argparse
import pprint

from data import Read



class Evaluate:
    def __init__(self, args) -> None:

        '''
        File-level evaluation (pred file)
        '''

        # print all args
        pprint.pprint(vars(args))

        self.pred = Read(in_file=args.pred_file,
                         language=args.language,
                         mode=args.mode,
                         is_pred=True)

        self.gold = Read(in_file=args.gold_file,
                         language=args.language,
                         mode=args.mode)

        assert len(self.pred.all_sentences) == len(self.gold.all_sentences), "Length mismatch"



    def UAS(self):

        correct_heads = 0
        total_tokens = 0

        for sent_idx in range(len(self.pred.all_sentences)):
            pred_heads = self.pred.all_sentences[sent_idx].head #list
            gold_heads = self.gold.all_sentences[sent_idx].head #list

            for token_idx in range(1, len(pred_heads)): # Skip ROOT
                total_tokens += 1
                if pred_heads[token_idx] == gold_heads[token_idx]:
                    correct_heads += 1
        
        return correct_heads / total_tokens



    def LAS(self):

        correct_heads_and_labels = 0
        total_tokens = 0

        for sent_idx in range(len(self.pred.all_sentences)):

            pred_heads = self.pred.all_sentences[sent_idx].head #list
            gold_heads = self.gold.all_sentences[sent_idx].head #list

            pred_rels = self.pred.all_sentences[sent_idx].rel #list
            gold_rels = self.gold.all_sentences[sent_idx].rel #list

            for token_idx in range(1, len(pred_heads)): # Skip ROOT

                total_tokens += 1

                if pred_heads[token_idx] == gold_heads[token_idx]:
                    if pred_rels[token_idx] == gold_rels[token_idx]:

                        correct_heads_and_labels += 1
        
        return correct_heads_and_labels / total_tokens



    def UCM(self):
        
        complete_matches = 0
        total_sentences = len(self.pred.all_sentences)

        for sent_idx in range(total_sentences):

            pred_heads = self.pred.all_sentences[sent_idx].head #list
            gold_heads = self.gold.all_sentences[sent_idx].head #list

            correct_heads = 0
            total_heads = 0

            for token_idx in range(1, len(pred_heads)): # Skip ROOT
                total_heads += 1
                if pred_heads[token_idx] == gold_heads[token_idx]:
                    correct_heads += 1
            
            if correct_heads == total_heads:
                complete_matches += 1
            
        
        return complete_matches / total_sentences
    



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

    parser.add_argument(
        "--metrics",
        type=str,
        default="uas,ucm",
        help="UAS, LAS, UCM",
    )


    args = parser.parse_args()

    evaluation = Evaluate(args)

    
    for metric in args.metrics.split(","):
        
        if metric == "uas":
            print(f"\nUAS: {round(evaluation.UAS(), 3)}\n")

        if metric == "las":
            print(f"LAS: {round(evaluation.LAS(), 3)}\n")
        
        if metric == "ucm":
            print(f"UCM: {round(evaluation.UCM(), 3)}\n")