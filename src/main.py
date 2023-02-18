
import pprint
import collections
import random

import os
import pickle
import gzip

import argparse

import time
from datetime import datetime

from data import Read
from features import Features
from graphs import Graph
from model import StructuredPerceptron



def main(args):

    # print all args
    pprint.pprint(vars(args))

    if args.mode == "train":

        _start_tr = time.time()

        # set data to use
        train_reader = Read(file_name=args.train_filename,
                            language=args.language,
                            mode=args.mode) 

        if args.train_slice == 0: # use all
            train_data = train_reader.all_sentences
        else:
            train_data = train_reader.all_sentences[:args.train_slice]

        # create feature map
        F = Features()
        feature_map = F.create_feature_map(train_data=train_data)

        # init weight vector
        weight_vector = init_w(n_dims=len(feature_map), init_type=args.init_type)

        print(f"\nSize of feature map & w: {len(feature_map)}")


        for epoch in range(1, args.n_epochs+1):

            _start_ep = time.time()

            print(f"\n+++ Epoch {epoch} +++")

            sent_num = 0
            all_uas = []

            for sentence in train_data:

                sent_num += 1 # First instance is sentence 1

                # init graphs
                gold_graph = Graph(sentence=sentence,
                                feature_map=feature_map,
                                weight_vector=weight_vector,
                                graph_type="gold").graph

                fully_connected_graph = Graph(sentence=sentence,
                                        feature_map=feature_map,
                                        weight_vector=weight_vector,
                                        graph_type="fully_connected").graph

                
                # call model
                model = StructuredPerceptron(weight_vector=weight_vector, 
                                    gold_graph=gold_graph, 
                                    fully_connected_graph=fully_connected_graph,
                                    feature_map=feature_map,
                                    mode=args.mode,
                                    lr=args.lr)

                # training iteration
                weight_vector, uas_sent, correct_arcs, total_arcs = model.train()
                
                all_uas.append(uas_sent)
                
                # print every n sentences
                if sent_num % args.print_every == 0:
                    
                    uas_so_far = sum(all_uas)/len(all_uas)
                    
                    # UAS general is avg of sentence UASs
                    print(f"Sent. {sent_num}, UAS sent: {round(uas_sent, 3)}, UAS general: {round(uas_so_far, 3)}")


            epoch_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-_start_ep))
            print(f"Epoch {epoch} time: {epoch_time}")


        if args.save_model:
            # save model + feature map
            save_model_fm(model=weight_vector, 
                            fm=feature_map, 
                            models_dir=args.models_dir)

        training_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-_start_tr))
        print(f"\nTotal training time: {training_time}")

    

    if args.mode == "dev" or args.mode == "test":

        _start = time.time()

        # set data to use
        test_reader = Read(file_name=args.test_filename,
                            language=args.language,
                            mode=args.mode)

        if args.test_slice == 0: # use all
            test_data = test_reader.all_sentences
        else:
            test_data = test_reader.all_sentences[:args.test_slice]


        # load model and feature map
        weight_vector, feature_map = load_model_fm(filename=args.model_filename,
                                                    models_dir=args.models_dir)
        
        # set filename for preds
        time_now = datetime.now().strftime("%d%m%H%M") #01021800

        out_file = os.path.join(args.preds_dir, 
            f"{args.language}-{args.mode}-{args.n_epochs}-{args.lr}-{args.init_type}",
            f"_{time_now}.conll06.pred"
        )

        tot_sentences = len(test_data)

        print(f"\nTotal sentences: {tot_sentences}",
            f"\nPredictions will be saved here: {out_file}\n")


        _start_pred = time.time()

        processed_sents = 0

        for sentence in test_data:

            # init graph
            global G
            G = Graph(sentence=sentence,
                        feature_map=feature_map,
                        weight_vector=weight_vector,
                        graph_type="fully_connected")

            fc_graph = G.graph
            
            # call model
            model = StructuredPerceptron(weight_vector=weight_vector, 
                                gold_graph=None, 
                                fully_connected_graph=fc_graph,
                                feature_map=feature_map,
                                mode=args.mode,
                                lr=args.lr)

            # make prediction 
            pred_graph = model.test()

            # write to file
            if pred_graph == {}:
                print("!! Prediction is empty")

            else:
                write_pred(sentence_ob=sentence, 
                            pred_graph=pred_graph, 
                            out_file=out_file)
                processed_sents += 1
                
            # print some info
            if processed_sents % args.print_every == 0:
                print(f"Sentences processed: {processed_sents}/{tot_sentences}")
        

        _end_pred = time.time()
        _time_elapsed = time.strftime("%H:%M:%S", time.gmtime(_end_pred-_start_pred))

        print(f"\nDone. Time elapsed: {_time_elapsed}"
              f"\nSentences processed: {processed_sents}/{tot_sentences}")



def write_pred(sentence_ob:object, pred_graph:dict, out_file:str):

    _rev = G.reverse_graph(graph=pred_graph)

    # order by int value of keys (correct order when printing)
    rev_pred_graph = collections.OrderedDict(
        {k: v for k, v in sorted(_rev.items(), key=lambda x: int(x[0]))}
    )


    with open(out_file, "a+") as f:

        for dep in rev_pred_graph.keys():

            head = list(rev_pred_graph[dep].keys())[0] # only one head

            dep_idx = int(dep)

            f.write(
                dep + "\t" +
                sentence_ob.form[dep_idx] + "\t" +
                sentence_ob.lemma[dep_idx] + "\t" +
                sentence_ob.pos[dep_idx] + "\t" +
                "_" + "\t" + # xpos
                "_" + "\t" + # morph
                head + "\t" +
                "_" + "\t" + # rel
                "_" + "\t" + # empty1
                "_" + "\n" # empty2
            )

        f.write("\n")



def init_w(n_dims:int, init_type:str) -> list:

    if init_type == "zeros":
        w = [0 for _ in range(n_dims)]

    if init_type == "random":
        lower = -1e-4
        upper = 1e-4
        w = [random.uniform(lower, upper) for _ in range(n_dims)]

    return w



def load_model_fm(filename:str, models_dir:str) -> list:
    
    in_file = os.path.join(models_dir, filename)
    f = gzip.open(in_file, "rb")

    model, fm = pickle.load(f)
    
    f.close()

    return model, fm



def save_model_fm(model:list, fm:dict, models_dir:str, args):
    
    time_now = datetime.now().strftime("%d%m%H%M") #01021800
    
    out_file = os.path.join(
        models_dir, 
        f"{args.language}-{args.n_epochs}-{args.lr}-{args.init_type}_", 
        time_now
    )

    print(f"Model filename:" {out_file})

    f = gzip.open(out_file, "wb")
    dump_obj = (model, fm)

    pickle.dump(dump_obj, f)
    f.close()





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train vs. dev/test",
    )

    parser.add_argument(
        "--n_epochs",
        type=int,
        default=3,
        help="number of training epochs",
    )

    parser.add_argument(
        "--init_type",
        type=str,
        default="zeros",
        help="init type for the weight vector (zeros vs. random)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.3,
        help="learning rate",
    )

    parser.add_argument(
        "--rand_seed",
        type=int,
        default=7,
        help="seed for random init of weight vector",
    )

    parser.add_argument(
        "--save_model",
        action='store_true',
        help="include --save_model to save the model, omit otherwise",
    )

    parser.add_argument(
        "--model_filename",
        type=str,
        default="eng-3ep-03_Tue14022023_1717",
        help="name of file to load weight vector and feature map from",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help="english data vs. german data",
    )

    parser.add_argument(
        "--train_filename",
        type=str,
        default="wsj_train.first-1k.conll06",
        help="filename of training data to be used (1k vs. 5k vs. full)",
    )

    parser.add_argument(
        "--train_slice",
        type=int,
        default=0,
        help="use only n instances for training (0: use all)",
    )

    parser.add_argument(
        "--test_filename",
        type=str,
        default="wsj_dev.conll06.blind",
        help="filename of test/dev data to be used (1k vs. 5k)",
    )

    parser.add_argument(
        "--test_slice",
        type=int,
        default=0,
        help="use only n instances for testing (0: use all)",
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=50,
        help="print UAS every n sentences",
    )

    parser.add_argument(
        "--models_dir",
        type=str,
        default="models/",
        help="directory where models are saved to/loaded from",
    )

    parser.add_argument(
        "--preds_dir",
        type=str,
        default="preds/",
        help="directory where predictions are saved",
    )


    args = parser.parse_args()
    main(args)