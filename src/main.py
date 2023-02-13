
import pprint
import collections
import random

import os
import pickle
import gzip

import argparse

import time
from datetime import datetime

from data import Read, Write, Sentence
from features import Features
from graph_v2 import Graph
from cle_v2 import Decoder
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
                    print(f"Sent. {sent_num}, UAS sent: {uas_sent}, UAS general: {uas_so_far}")


            epoch_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-_start_ep))
            print(f"Epoch {epoch} time: {epoch_time}")


        # save model + feature map
        save_model_fm(model=weight_vector, 
                        fm=feature_map, 
                        models_dir=args.models_dir)

        training_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-_start_tr))
        print(f"\nTotal training time: {training_time}")

    

    if args.mode == "dev" or args.mode == "test":

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
        

        uas_total = 0
        num_sentences = len(test_data)

        for sentence in test_data:

            # init graph
            fully_connected_graph = Graph(sentence=sentence,
                                    feature_map=feature_map,
                                    weight_vector=weight_vector,
                                    graph_type="fully_connected").graph
            
            # call model
            model = StructuredPerceptron(weight_vector=weight_vector, 
                                gold_graph=None, 
                                fully_connected_graph=fully_connected_graph,
                                feature_map=feature_map,
                                mode=args.mode,
                                lr=args.lr)

            # make prediction 
            pred_grap, uas_sent, correct_arcs, total_arcs = model.test()

            uas_total += uas_sent
        
        
        uas_general = uas_total / num_sentences
        print(f"Done. UAS: {uas_general}")
        
        # write preds to file



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



def save_model_fm(model:list, fm:dict, models_dir:str):
    
    time_now = datetime.now().strftime("%a%d%m%Y_%H%M") #Wed01022023_1800
    out_file = os.path.join(models_dir, time_now)

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
        "--model_filename",
        type=str,
        default="",
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
        help="filename of training data to be used (1k vs. 5k)",
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

    args = parser.parse_args()
    main(args)