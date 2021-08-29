import os
import utils
import torch
import random
import pickle
import logging
import argparse

import numpy as np
import torch.nn as nn

import model.learner as learner
import model.modelfactory as mf
import datasets.datasetfactory as df
import datasets.miniimagenet as imgnet

from scipy import stats
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from experiment.experiment import experiment

def load(file):
    openFile = open(file, "rb")
    data = pickle.load(openFile)
    openFile.close()
    return(data)

logger = logging.getLogger('experiment')

def pickle_struct(dictionary, filename): 
    p = pickle.Pickler(open("{0}.p".format(filename),"wb")) 
    p.fast = True 
    p.dump(dictionary) 

def train_imagenet(model, optimizer, classes, offset, device, updates):

    regulation = {t:[] for t in range(100)}
    imagenetClasses = np.random.choice(range(100), 100, replace=False)
    
    for c in imagenetClasses:

        counter = 0
        chosen = range(600)
        if c < 64:

            imagenet = imgnet.MiniImagenet(
                args.imagenet_path,
                mode='train',
                elem_per_class=chosen,#300,#updates,
                test=False,
                classes=[c])

        elif c > 63:

            if c > 83:

                imagenet = imgnet.MiniImagenet(
                    args.imagenet_path,
                    mode='val',
                    elem_per_class=chosen,#300,#updates,
                    test=False,
                    classes=[c-84])

            else:

                imagenet = imgnet.MiniImagenet(
                    args.imagenet_path,
                    mode='test',
                    elem_per_class=chosen,#300,#updates,
                    test=False,
                    classes=[c-offset])

        sorted_imagenet = torch.utils.data.DataLoader(imagenet, batch_size=1,
                               shuffle=args.iid, num_workers=1)

        for img, y in sorted_imagenet:


            if c > 63:
                if c > 83:
                    y += 84
                else:
                    y += offset

            task_key = y.item()

            img = img.to(device)
            y = y.to(device)

            pred, fw, w = model(img, analysis=True, layer_to_record=args.layer_to_record)
            regulation[c].append(np.mean(fw/w))

    pickle_struct(regulation, "Mean_Regulation_All_Images_Seed={0}".format(args.model_seed))
           
def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)

    frozen_layers = []
    for temp in range(args.rln * 2):
        frozen_layers.append("vars." + str(temp))

    logger.info("Frozen layers = %s", " ".join(frozen_layers))

    performance = []
    final_results_all = []
    temp_result = []
    total_clases = args.schedule

    for tot_class in total_clases:

        trials = 50
        lr_list = [0.0002]
        lr_all = []

        for lr_search in range(trials):

            print(args)

            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            results_mem_size = {}

            for mem_size in [args.memory]:

                max_acc = -10
                max_lr = -10

                for lr in lr_list:

                    maml = torch.load(args.model, map_location='cpu')
                    if args.scratch:
                        config = mf.ModelFactory.get_model("MRCL", args.dataset)
                        maml = learner.Learner(config)

                    maml = maml.to(device)

                    for name, param in maml.named_parameters():
                        if name in frozen_layers:
                            param.learn = False
                        else:
                            if args.reset:
                                w = nn.Parameter(torch.ones_like(param))
                                if len(w.shape) > 1:
                                    torch.nn.init.kaiming_normal_(w)
                                else:
                                    w = nn.Parameter(torch.zeros_like(param))
                                param.data = w
                                param.learn = True
                            else:
                                param.learn = True

                    frozen_layers = []
                    for temp in range(args.rln * 2):
                        frozen_layers.append("vars." + str(temp))

                    torch.nn.init.kaiming_normal_(maml.parameters()[-2])
                    w = nn.Parameter(torch.zeros_like(maml.parameters()[-1]))
                    maml.parameters()[-1].data = w

                    for n, a in maml.named_parameters():
                        n = n.replace(".", "_")
                       
                        if n == "vars_18" or n == "vars_32":

                            if n == "vars_32":
                                new_size = [1000,112]
                            elif n == "vars_18":
                                new_size = [1000*112, 1728]
                            else:
                                new_size = a.size()

                            w = nn.Parameter(torch.ones(new_size)).to('cuda')
                            torch.nn.init.kaiming_normal_(w)
                            a.data = w

                        if n == "vars_19" or n == "vars_33":

                            if n == "vars_33":
                                new_size = [1000]

                            if n == "vars_19":
                                new_size = [112*1000]

                                bias_init = args.bias
                                w = nn.Parameter(torch.zeros(new_size)).to('cuda')
                                w.data.fill_(bias_init)
                                a.data = w

                            else:
                                w = nn.Parameter(torch.zeros(new_size)).to('cuda')
                                a.data = w
 
                    filter_list = ["vars.{0}".format(v) for v in range(6)]

                    logger.info("Filter list = %s", ",".join(filter_list))
                    list_of_names = list(
                        map(lambda x: x[1], list(filter(lambda x: x[0] not in filter_list, maml.named_parameters()))))

                    list_of_params = list(filter(lambda x: x.learn, maml.parameters()))
                    list_of_names = list(filter(lambda x: x[1].learn, maml.named_parameters()))
                    if args.scratch or args.no_freeze:
                        print("Empty filter list")
                        list_of_params = maml.parameters()
                    
                    for x in list_of_names:
                        logger.info("Unfrozen layer = %s", str(x[0]))

                    opt = torch.optim.Adam(list_of_params, lr=lr)

                    classes = np.random.choice(range(100), 100, replace=False)

                    if args.dataset == "cifar":
                        cifar_data = utils.remove_classes(
                            df.DatasetFactory.get_dataset("CIFAR100", train=True), classes)

                        sorted_cifar = torch.utils.data.DataLoader(
                            utils.iterator_sorter(cifar_data, False, classes=classes),
                            batch_size=1,
                            shuffle=args.iid, num_workers=1)


                        train_cifar(sorted_cifar, maml, opt, args.num_updates, device)

                    elif args.dataset == "imagenet":
                        train_imagenet(maml, opt, tot_class, 64, device, args.num_updates)

                    exit()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_updates', type=int, help='Number of updates', default=20)
    argparser.add_argument('--run', type=int, help='run number', default=0)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1)
    argparser.add_argument('--layer_to_record', type=str, help='layer whose data to record', default='C3')
    argparser.add_argument('--treatment', type=str, help='which treatment', default='Grow')

    argparser.add_argument('--model_seed', type=int, help='seed used to meta-train', default=0)
    argparser.add_argument('--seed', type=int, help='epoch number', default=222)
    argparser.add_argument('--schedule', type=int, nargs='+', default=[100],
                        help='Decrease learning rate at these epochs.')
    argparser.add_argument('--memory', type=int, help='epoch number', default=0)
    argparser.add_argument('--model', type=str, help='epoch number', default="none")
    argparser.add_argument('--scratch', action='store_true', default=False)
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument('--dataset-path', help='Name of experiment', default=None)
    argparser.add_argument('--name', help='Name of experiment', default="evaluation")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-freeze", action="store_true")
    argparser.add_argument('--reset', action="store_true")
    argparser.add_argument('--test', action="store_true")
    argparser.add_argument("--iid", action="store_true")
    argparser.add_argument("--rln", type=int, default=6)
    argparser.add_argument("--runs", type=int, default=50)
    argparser.add_argument("--neuromodulation", action="store_true")
    argparser.add_argument("--bias", type=float, default=0.0)
    argparser.add_argument("--save_name", type=str, default="performance")
    argparser.add_argument('--imagenet_path', help='Name of experiment', default="/users/s/b/sbeaulie/meta-learning_neuromodulation_for_catastrophic_forgetting/anml/imagenet_data/")


    args = argparser.parse_args()

    import os

    args.name = "/".join([args.dataset, "eval", str(args.epoch).replace(".", "_"), args.name])

    main(args)
