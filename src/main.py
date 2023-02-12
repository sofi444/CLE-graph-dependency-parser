
import pprint
import collections
import random

import os
import pickle
import gzip

import argparse

from datetime import datetime

from data import Read, Write, Sentence
from features import Features
from graph_v2 import Graph
from cle_v2 import Decoder
from model import StructuredPerceptron



def main(args):

    if args.mode == "train":

        train_reader = Read(file_name="wsj_train.first-1k.conll06",
                            language="english",
                            mode="train") 


        train_data = train_reader.all_sentences
        #train_data = train_reader.all_sentences[:100]

        # create feature map
        F = Features()
        feature_map = F.create_feature_map(train_data=train_data)

        weight_vector = init_w(n_dims=len(feature_map), init_type=args.init_type)

        eval_dict = {epoch_num: 
                        {sent_num: 0.0 # sentence-level UAS
                        for sent_num in range(1, len(train_data)+1)
                        } 
                    for epoch_num in range(1, args.n_epochs+1)
                    }

        for epoch in range(1, args.n_epochs+1):

            print(f"+++ Epoch {epoch} +++")

            sent_num = 0

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
                
                # populate eval dict
                eval_dict[epoch][sent_num] = uas_sent
                

                if sent_num % 50 == 0:
                #if sent_num % 100 == 0:

                    _sents_so_far = sent_num
                    
                    uas_so_far = sum(
                        list(eval_dict[epoch].values())
                    ) / _sents_so_far
                    
                    print(f"Sent. {sent_num}, UAS sent: {uas_sent}, UAS general: {uas_so_far}")


        # save model + feature map
        save_model_fm(weight_vector, feature_map)

    

    if args.mode == "dev" or args.mode == "test":

        test_reader = Read(file_name="wsj_dev.conll06.blind",
                            language="english",
                            mode="dev")

        test_data = test_reader.all_sentences[:5]

        weight_vector, feature_map = load_model_fm(filename=args.model_filename)

        uas_total = 0
        num_sentences = len(test_data)

        for sentence in test_data:

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



def load_model_fm(filename:str, models_dir="models/") -> list:
    
    in_file = os.path.join(models_dir, filename)
    f = gzip.open(in_file, "rb")

    model, fm = pickle.load(f)
    
    f.close()

    return model, fm



def save_model_fm(model:list, fm:dict, models_dir="models/"):
    
    time_now = datetime.now().strftime("%a%d%m%Y_%H%M") #Wed01022023_1800
    out_file = os.path.join(models_dir, time_now)

    f = gzip.open(out_file, "wb")
    dump_obj = (model, fm)

    pickle.dump(dump_obj, f)
    f.close()



"""

# will be args:
#mode = "train"
#mode = "dev"
#n_epochs = 3
#init_type = "zeros"
#lr = 0.3
#model_filename = "Thu02022023_2000"


test_reader = Read(file_name="wsj_dev.conll06.blind",
                    language="english",
                    mode="dev")

train_reader = Read(file_name="wsj_train.first-1k.conll06",
                    language="english",
                    mode="train") 


train_data = train_reader.all_sentences
#train_data = train_reader.all_sentences[:100]
test_data = test_reader.all_sentences[:5]



if mode == "train":

    # create feature map
    F = Features()
    feature_map = F.create_feature_map(train_data=train_data)

    weight_vector = init_w(n_dims=len(feature_map), init_type=init_type)

    eval_dict = {epoch_num: 
                    {sent_num: 0.0 # sentence-level UAS
                    for sent_num in range(1, len(train_data)+1)
                    } 
                for epoch_num in range(1, n_epochs+1)
                }

    for epoch in range(1, n_epochs+1):

        print(f"+++ Epoch {epoch} +++")

        sent_num = 0

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
                                mode=mode,
                                lr=lr)

            # training iteration
            weight_vector, uas_sent, correct_arcs, total_arcs = model.train()
            
            # populate eval dict
            eval_dict[epoch][sent_num] = uas_sent
            

            if sent_num % 50 == 0:
            #if sent_num % 100 == 0:

                _sents_so_far = sent_num
                
                uas_so_far = sum(
                    list(eval_dict[epoch].values())
                ) / _sents_so_far
                
                print(f"Sent. {sent_num}, UAS sent: {uas_sent}, UAS general: {uas_so_far}")


    # save model + feature map
    save_model_fm(weight_vector, feature_map)



if mode == "dev" or mode == "test":

    weight_vector, feature_map = load_model_fm(filename=model_filename)

    uas_total = 0
    num_sentences = len(test_data)

    for sentence in test_data:

        fully_connected_graph = Graph(sentence=sentence,
                                feature_map=feature_map,
                                weight_vector=weight_vector,
                                graph_type="fully_connected").graph
        
        # call model
        model = StructuredPerceptron(weight_vector=weight_vector, 
                            gold_graph=None, 
                            fully_connected_graph=fully_connected_graph,
                            feature_map=feature_map,
                            mode=mode,
                            lr=lr)

        pred_grap, uas_sent, correct_arcs, total_arcs = model.test()

        uas_total += uas_sent
    
    
    uas_general = uas_total / num_sentences
    print(f"Done. UAS: {uas_general}")
    
    
    # write preds to file

"""


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


    args = parser.parse_args()
    main(args)