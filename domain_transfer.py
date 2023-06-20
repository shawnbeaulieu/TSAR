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


def Entropy(P, bins):

   try:
       E = -np.sum(P*np.log2(P)/np.log2(bins), axis=1)
   except:
       print("Single Vector")
       E = -np.sum(P*np.log2(P)/np.log2(bins))

   return(E)


def PMI(cooccurrence_matrix, ppmi=False):

    mutual = np.zeros_like(cooccurrence_matrix.astype(float))
    freq = cooccurrence_matrix/3000
    for i in range(cooccurrence_matrix.shape[0]):
        for j in range(cooccurrence_matrix.shape[1]):
            if i == j:
                pmi = 0
            else:
                x = freq[i,i]
                y = freq[j,j]
                xy = freq[i,j]
                if xy == 0:
                    pmi = 0
                else:
                    pmi = np.log2(xy/(x*y))
                    norm = -np.log2(xy)
                    pmi = pmi/norm

                if ppmi:
                    mutual[i,j] = np.max([pmi, 0])
                else:
                    mutual[i,j] = pmi

    return(mutual)

def compute_timelag(signals, bins=250):

    centile = int((112*112*3*3)*0.01)
    indices = np.argsort(np.mean(signals, axis=0))
    timelag = np.zeros((bins,bins))
    for it in range(1,3000):
        timelag += np.histogram2d(y=signals[it,indices[centile*75:centile*99]], 
                                  x=signals[it-1,indices[centile*75:centile*99]], 
                                  range=[[0,1],[0,1]], bins=bins)[0]

    np.save("Timelag_Scarce_75to99_Grow_Layer={0}_Seed={1}_Dataset={2}".format(args.layer_to_record,
                                                                               args.model_seed, 
                                                                               args.dataset), 
                                                                               timelag)


def compute_fc_past(functional_weights, weights, task_order):

    signals = []
    for t in np.arange(0,100,1):
        w = np.array(weights[task_order[t]])[:,:100,:]
        fw = np.array(functional_weights[task_order[t]])[:,:100,:]
        signals.append(fw/w)
    signals = np.array(signals)

    current_p = []
    prior_p = []
    current_n = []
    prior_n = []

    for t in range(5,95):

        pindices = np.argwhere(w[-1,task_order[t],:] >= 0)
        nindices = np.argwhere(w[-1,task_order[t],:] < 0)

        current_p.append(list(np.transpose(np.mean(signals[t,:,task_order[t],pindices], axis=0))))
        prior_p.append(list(np.transpose(np.mean(signals[t-5,:,task_order[t],pindices], axis=0))))
        current_n.append(list(np.transpose(np.mean(signals[t,:,task_order[t],nindices], axis=0))))
        prior_n.append(list(np.transpose(np.mean(signals[t-5,:,task_order[t],nindices], axis=0))))


    current_n = np.array(current_n).reshape(90,-1)
    prior_n = np.array(prior_n).reshape(90,-1)

    current_p = np.array(current_p).reshape(90,-1)
    prior_p = np.array(prior_p).reshape(90,-1)

    current_change_n = ((prior_n[:,-1].reshape(90,1) - current_n)/prior_n[:,-1].reshape(90,1))*100
    current_change_p = ((prior_p[:,-1].reshape(90,1) - current_p)/prior_p[:,-1].reshape(90,1))*100

    np.save("current_change_n_Seed={0}".format(args.model_seed), current_change_n)
    np.save("current_change_p_Seed={0}".format(args.model_seed), current_change_p)

def compute_fc_future(functional_weights, weights, task_order):

    signals = []
    for t in np.arange(0,100,1):
        w = np.array(weights[task_order[t]])[:,:100,:]
        fw = np.array(functional_weights[task_order[t]])[:,:100,:]
        signals.append(fw/w)
    signals = np.array(signals)

    current_p = []
    prior_p = []
    future_p = []
    current_n = []
    prior_n = []
    future_n = []
    for t in range(10,90):
        pindices = np.argwhere(w[-1,task_order[t],:] >= 0)
        nindices = np.argwhere(w[-1,task_order[t],:] < 0)
        current_p.append(list(np.transpose(np.mean(signals[t,:,task_order[t],pindices], axis=0))))
        current_n.append(list(np.transpose(np.mean(signals[t,:,task_order[t],nindices], axis=0))))
        temp_p = []
        temp_n = []
        for shift in range(1,11):
            temp_p.append(list(np.transpose(np.mean(signals[t+shift,:,task_order[t],pindices], axis=0))))
            temp_n.append(list(np.transpose(np.mean(signals[t+shift,:,task_order[t],nindices], axis=0))))
        future_p.append(temp_p)
        future_n.append(temp_n)

    current_n = np.array(current_n).reshape(80,30)
    future_n = np.array(future_n).reshape(80,300)
    current_p = np.array(current_p).reshape(80,30)
    future_p = np.array(future_p).reshape(80,300)
    future_change_p = ((current_p[:,-1].reshape(80,1) - future_p)/current_p[:,-1].reshape(80,1))*100
    future_change_n = ((current_n[:,-1].reshape(80,1) - future_n)/current_n[:,-1].reshape(80,1))*100

    np.save("future_change_n_Seed={0}".format(args.model_seed), future_change_n)
    np.save("future_change_p_Seed={0}".format(args.model_seed), future_change_p)

def compute_rank_rank_histo(signals):

    signals = signals.reshape(100, 30, -1)
    upper_bound = signals.shape[2]
    countMatrix = np.zeros((100,100))

    for task in range(100):

        idex = np.array(range(task+1,task+100))%100
        global_rank = (signals.shape[2] - rankdata(np.mean(signals[idex, :, :].reshape(99*30, -1), axis=0))) + 1
        local_rank = (signals.shape[2] - rankdata(np.mean(signals[task,:,:], axis=0))) + 1
        countMatrix += np.histogram2d(x=np.log10(global_rank), 
                                      y=np.log10(local_rank), 
                                      bins=100, 
                                      range=[[0,np.log10(upper_bound)],[0,np.log10(upper_bound)]])

    np.save("TSAR_Scarce_rankrank_Layer={0}_Model={1}_Seed={2}_Dataset={3}.npy".format(args.layer_to_record, args.model_seed, args.seed, args.dataset), countMatrix[0])

def compute_avalanches(signals):

    signals = signals.reshape(3000, -1)
    for threshold in [0.10, 0.25, 0.50]:
        spikes = np.sum((signals > threshold).astype('uint8'), axis=1)
        avalanche = spikes
        histo = np.histogram(avalanche, range=(0, signals.shape[1]), bins=2000)
        np.save("Avalanche_Histogram_Layer={0}_Thresh={1}_Run={2}_Dataset={3}_Model={4}".format(args.layer_to_record, threshold, args.seed, args.dataset, args.model_seed), histo)


def compute_ppmi(c2, c3, model_seed):

    c2 = c2.reshape(3000,-1)
    c3 = c3.reshape(3000,-1)

    all_mutual = {}
    for threshold in [0.10, 0.25, 0.5]:
        c2_spikes = (c2 > threshold).astype('uint8')
        c2_firing_rate = np.mean(c2_spikes, axis=0)
        c2_indices = np.where(c2_firing_rate > 0.01)[0]

        c3_spikes = (c3 > threshold).astype('uint8')
        c3_firing_rate = np.mean(c3_spikes, axis=0)
        c3_indices = np.where(c3_firing_rate > 0.01)[0]

        spikes = np.concatenate([c2_spikes[:, c2_indices], c3_spikes[:, c3_indices]], axis=1)
        cooccurrence_matrix = np.dot(spikes.transpose().astype(int), spikes.astype(int))
        mutual = np.zeros_like(cooccurrence_matrix.astype(float))
        freq = cooccurrence_matrix/3000

        for i in range(cooccurrence_matrix.shape[0]):
            for j in range(cooccurrence_matrix.shape[1]):
                x = freq[i,i]
                y = freq[j,j]
                xy = freq[i,j]
                if xy == 0:
                    pmi = 0
                else:
                    pmi = np.log2(xy/(x*y))
                    norm = -1.0*np.log2(xy)
                    pmi = pmi/norm
                mutual[i,j] = pmi

        m = np.clip(mutual, 0, 1)
        c2_to_c3 = m[:c2_indices.shape[0], c2_indices.shape[0]:]
        #upper_triangle = m[np.triu_indices(m.shape[0], 1)]
        histo = np.histogram(c2_to_c3, range=(0,np.max(c2_to_c3)), bins=2000)
        all_mutual[threshold] = histo

    np.save("C2C3_Mutual_Seed={0}".format(model_seed), all_mutual)

def train_omniglot(iterator, model, optimizer, device):

    for img, y in iterator:
        img = img.to(device)
        y = y.to(device)
        task_key = y.item()

        pred = model(img, ANML=args.ANML, analysis=False)
        optimizer.zero_grad()
        loss = F.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()

def train_imagenet(model, optimizer, imagenetClasses, offset, device, updates, train_chosen, test_chosen):

    history = []
    performance = []
    signals = []
    counter = 0
    for c in imagenetClasses:

        # 64 "training" classes + 10 "validation" classes + 16 "test" classes
        # For domain transfer, combine all to get 100 total classes:

        history.append(c)
        if c < 64:
            imagenet = imgnet.MiniImagenet(
                args.imagenet_path,
                mode='train',
                elem_per_class=train_chosen[c],#updates,
                test=False,
                classes=[c])
        elif c > 63:
            if c > 83:
                imagenet = imgnet.MiniImagenet(
                    args.imagenet_path,
                    mode='val',
                    elem_per_class=train_chosen[c],#updates,
                    test=False,
                    classes=[c-84])
            else:
                imagenet = imgnet.MiniImagenet(
                    args.imagenet_path,
                    mode='test',
                    elem_per_class=train_chosen[c],#updates,
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
            if args.analysis:
                pred, fw, w = model(img, analysis=True, layer_to_record=args.layer_to_record)
                signals.append((fw/w).reshape(1,-1))

            else:
                pred = model(img, ANML=args.ANML)
            optimizer.zero_grad()
            loss = F.cross_entropy(pred, y)
            loss.backward()
            optimizer.step()

        counter += 1

        if args.imagenet_condition == 'random':

            if counter in [100]:
                if args.test:
                    performance.append(evaluate_imagenet(model, 
                                                         history, 
                                                         offset, 
                                                         device, 
                                                         updates, 
                                                         test_chosen, 
                                                         test=True))
                else:
                    performance.append(evaluate_imagenet(model, 
                                                         history, 
                                                         offset, 
                                                         device, 
                                                         updates, 
                                                         train_chosen, 
                                                         test=False))
          
        else:

            if counter in [100]:
                if args.test:
                    performance.append(evaluate_imagenet(model, 
                                                         history, 
                                                         offset, 
                                                         device, 
                                                         updates, 
                                                         test_chosen, 
                                                         test=True))
                else:
                    performance.append(evaluate_imagenet(model, 
                                                         history, 
                                                         offset, 
                                                         device, 
                                                         updates, 
                                                         train_chosen, 
                                                         test=False))

    if args.analysis:

        signals = np.array(signals).reshape(3000,-1)
        counts = np.histogram(np.array(signals).flatten(), range=(0,1+1/1000), bins=1000)
        np.save("{0}_SignalCounts_Layer={1}_ImageNet_Model={2}".format(args.model, 
                                                                       args.layer_to_record, 
                                                                       args.model_seed), 
                                                                       counts)
        compute_avalanches(signals)

    return(performance)

def train_cifar(iterator, model, optimizer, num_updates, device):

    iteration = 0
    total_count = 0

    for img, y in iterator:

        if iteration == 0:
            current_class = y

        if iteration >= num_updates:
            if current_class != y:
                iteration = 1
                current_class = y
                img = img.to(device)
                y = y.to(device)
                pred = model(img, ANML=args.ANML)
                optimizer.zero_grad()
                loss = F.cross_entropy(pred, y)
                loss.backward()
                optimizer.step()
                total_count += 1

            else:
                next

        else:
            img = img.to(device)
            y = y.to(device)
            pred = model(img, ANML=args.ANML)
            optimizer.zero_grad()
            loss = F.cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
            iteration += 1
            total_count += 1

def evaluate_omniglot(iterator, model, device):

    correct = 0
    for img, y in iterator:

        img = img.to(device)
        y = y.to(device)
        logits_q = model(img, vars=None, bn_training=False, ANML=args.ANML, feature=False)

        pred_q = (logits_q).argmax(dim=1)
        correct += torch.eq(pred_q, y).sum().item() / len(img)

    return(correct/len(iterator))

def evaluate_imagenet(model, imagenetClasses, offset, device, updates, chosen, test):

    correct = 0
    counter = 0
    past_performance = []

    for c in imagenetClasses:

        if c < 64:

            imagenet = imgnet.MiniImagenet(
                args.imagenet_path,
                mode='train',
                elem_per_class=chosen[c],#updates,
                test=test,
                classes=[c])

        elif c > 63:

            if c > 83:

                imagenet = imgnet.MiniImagenet(
                    args.imagenet_path,
                    mode='val',
                    elem_per_class=chosen[c],#updates,
                    test=test,
                    classes=[c-84])

            else:

                imagenet = imgnet.MiniImagenet(
                    args.imagenet_path,
                    mode='test',
                    elem_per_class=chosen[c],#updates,
                    test=test,
                    classes=[c-offset])

        sorted_imagenet = torch.utils.data.DataLoader(imagenet, batch_size=1,
                               shuffle=args.iid, num_workers=1)


        for img, y in sorted_imagenet:
    
            if c > 63:
                if c > 83:
                    y += 84
                else:
                    y += offset

            current_task = y.item()
     
            img = img.to(device)
            y = y.to(device)
            logits_q = model(img, vars=None, ANML=args.ANML)
            pred_q = (logits_q).argmax(dim=1)
            correct += float(torch.eq(pred_q, y).sum().item() / len(img))

        counter += 1

    return(correct/(len(imagenetClasses)*updates))

def evaluate_cifar(iterator, model, device, num_updates):

    correct = 0
    iteration = 0
    total_count = 0

    for img, y in iterator:

        if iteration == 0:
            current_class = y

        if iteration >= num_updates:

            if current_class != y:
                iteration = 1
                current_class = y
                img = img.to(device)
                y = y.to(device)
                logits_q = model(img, vars=None, bn_training=False, feature=False, ANML=args.ANML)

                pred_q = (logits_q).argmax(dim=1)
                correct += torch.eq(pred_q, y).sum().item() / len(img)
                total_count += 1

            else:

                next

        else:

            img = img.to(device)
            y = y.to(device)
            logits_q = model(img, vars=None, bn_training=False, feature=False, ANML=args.ANML)

            pred_q = (logits_q).argmax(dim=1)
            correct += torch.eq(pred_q, y).sum().item() / len(img)
            iteration += 1
            total_count += 1

    return(correct/total_count)


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
    print(args)

    for tot_class in total_clases:

        trials = args.num_trials
        lr_list = [0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
        lr_all = []

        for lr_search in range(trials):

            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            results_mem_size = {}

            for mem_size in [args.memory]:

                max_acc = -10
                max_lr = -10

                for lr in lr_list:

                    print("Starting domain transfer with learning rate={0}".format(lr))

                    # ============= LOAD and PREPARE Model =============

                    maml = torch.load("{0}={1}".format(args.model, args.model_seed), 
                            map_location='cpu') 

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

                    # ============= Fixing and/or resetting parameters =============

                    for n, a in maml.named_parameters():
                        n = n.replace(".", "_")
                       
                        if args.ANML:

                           if n == "vars_26":
                               new_size = a.size()
                               w = nn.Parameter(torch.ones(new_size)).to('cuda')
                               torch.nn.init.kaiming_normal_(w)
                               a.data = w
                           if n == "vars_27":
                                new_size = a.size()
                                w = nn.Parameter(torch.zeros(new_size)).to('cuda')
                                a.data = w 

                        else:
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
                                    new_size = a.size()
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

                    # ============= LOADING domain transfer datasets =============


                    if args.dataset == 'omniglot' or args.dataset == 'omniglot_test':

                        omniglot_keep = np.array(range(600))

                        omniglot_data = utils.remove_classes_omni(
                            df.DatasetFactory.get_dataset("omniglot", train=True, background=False),
                            omniglot_keep)

                        omniglot_test_data = utils.remove_classes_omni(
                            df.DatasetFactory.get_dataset("omniglot", train=False, background=False),
                            omniglot_keep)

                        sorted_omniglot = torch.utils.data.DataLoader(
                            utils.iterator_sorter_omni(omniglot_data, False, classes=600),
                            batch_size=1,
                            shuffle=args.iid, num_workers=1)

                        omniglot_test = torch.utils.data.DataLoader(
                            utils.iterator_sorter_omni(omniglot_test_data, False, classes=600),
                            batch_size=1,
                            shuffle=args.iid, num_workers=1)

                        train_omniglot(sorted_omniglot, maml, opt, device)

                        if args.test:
                            accuracy = evaluate_omniglot(omniglot_test, maml, device)
                        else:
                            accuracy = evaluate_omniglot(sorted_omniglot, maml, device)

                    if args.dataset == 'cifar':

                        classes = np.random.choice(range(100), 100, replace=False)

                        cifar_data = utils.remove_classes(
                            df.DatasetFactory.get_dataset("CIFAR100", train=True), classes)

                        sorted_cifar = torch.utils.data.DataLoader(
                            utils.iterator_sorter(cifar_data, False, classes=classes),
                            batch_size=1,
                            shuffle=args.iid, num_workers=1)

                        train_cifar(sorted_cifar, maml, opt, args.num_updates, device)
                        accuracy = evaluate_cifar(sorted_cifar, maml, device, args.num_updates)
                    
                    elif args.dataset == 'imagenet':

                        classes = np.random.choice(range(100), 100, replace=False)
                        if args.imagenet_condition != 'random':
                            class2meanreg = \
                                load("Mean_Regulation_All_Images_Seed={0}.p".format(args.model_seed))

                        train_chosen = {}
                        test_chosen = {}
                        counter = 0

                        for cl in classes:

                            if args.imagenet_condition == 'random':
                                if args.test:
                                    train_chosen[cl] = np.random.choice(range(300,600), 30, replace=False)
                                    test_chosen[cl] = np.random.choice(range(300), 30, replace=False)

                                else:
                                    train_chosen[cl] = np.random.choice(range(600), 30, replace=False)
               
                            elif args.imagenet_condition == 'spikes':
                                sorted_by_reg = np.argsort(class2meanreg[cl])
                                spikes = sorted_by_reg[-30:]
                                rand_idx = np.random.choice(range(30), 30, replace=False)
                                train_chosen[cl] = spikes[rand_idx]

                            elif args.imagenet_condition == 'nonspikes':
                                sorted_by_reg = np.argsort(class2meanreg[cl])
                                nonspikes = sorted_by_reg[:30]
                                rand_idx = np.random.choice(range(30), 30, replace=False)
                                train_chosen[cl] = nonspikes[rand_idx]

                            elif args.imagenet_condition == 'mixed':

                                sorted_by_reg = np.argsort(class2meanreg[cl])
                                rand_idx = np.random.choice(range(30), 30, replace=False)
                                diminishing = sorted_by_reg[:30]
                                enhancing = sorted_by_reg[-30:]
                                rand_spike = np.random.choice(range(len(enhancing)), args.num_spikes, replace=False)     
                                rand_nospike = np.random.choice(range(len(diminishing)), 30-args.num_spikes, replace=False)
                                instances = np.concatenate([enhancing[rand_spike], diminishing[rand_nospike]])
                                np.random.shuffle(instances)
                                train_chosen[cl] = instances

                        if args.test:
                            accuracy = train_imagenet(maml, 
                                                      opt, 
                                                      classes, 
                                                      64, 
                                                      device, 
                                                      30, 
                                                      train_chosen, 
                                                      test_chosen)

                            performance.append(accuracy)
                            accuracy = accuracy[-1]

                        else:
                            accuracy = train_imagenet(maml, 
                                                      opt, 
                                                      classes, 
                                                      64, 
                                                      device, 
                                                      30, 
                                                      train_chosen, 
                                                      train_chosen)

                            performance.append(accuracy)
                            accuracy = accuracy[-1]

                    print(performance)

                    # ============= RECORD performance =============

                    if len(lr_list) == 1:
                        if args.dataset != 'imagenet':
                            performance.append(accuracy)
                        np.save("{0}={1}_Dataset={2}".format(args.model, 
                                                             args.model_seed, 
                                                             args.dataset), performance)

                    logger.info("Result after one epoch for LR = %f", lr)
                    logger.info(str(accuracy))

                    if (accuracy > max_acc):
                        max_acc = accuracy
                        max_lr = lr

                print("Accuracy for LR={0}: {1}".format(lr, accuracy))
                lr_all.append(max_lr)
                results_mem_size[mem_size] = (max_acc, max_lr)
                logger.info("Final Max Result = %s", str(max_acc))

            temp_result.append((tot_class, results_mem_size))
            print("A=  ", results_mem_size)
            logger.info("Temp Results = %s", str(results_mem_size))

            print("LR RESULTS = ", temp_result)

            best_lr = float(stats.mode(lr_all)[0][0])
            logger.info("BEST LR %s= ", str(best_lr))
            lr_list = [best_lr]

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_updates', type=int, help='Number of updates', default=20)
    argparser.add_argument('--num_spikes', type=int, help='Number of spikes', default=15)
    argparser.add_argument('--random_before', type=int, help='Number of spikes', default=15)
    argparser.add_argument('--run', type=int, help='run number', default=0)
    argparser.add_argument('--analysis', action='store_true', default=False)
    argparser.add_argument('--imagenet_condition', type=str,default='random')
    argparser.add_argument('--epochs', type=int, help='epoch number', default=1)
    argparser.add_argument('--seed', type=int, help='epoch number', default=222)
    argparser.add_argument('--model_seed', type=int, help='seed model was trained with', default=222)
    argparser.add_argument('--treatment', type=str, help='which treatment', default='Grow')
    argparser.add_argument('--condition', type=str, help='rich or scarce', default='Rich')
    argparser.add_argument('--schedule', type=int, nargs='+', default=[600],
                        help='Decrease learning rate at these epochs.')
    argparser.add_argument('--memory', type=int, help='epoch number', default=0)
    argparser.add_argument('--num_trials', type=int, help='Number of trials', default=50)
    argparser.add_argument('--model', type=str, help='epoch number', default="none")
    argparser.add_argument('--scratch', action='store_true', default=False)
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument('--dataset-path', help='Name of experiment', default=None)
    argparser.add_argument('--name', help='Name of experiment', default="evaluation")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--ANML", action="store_true")
    argparser.add_argument("--no-freeze", action="store_true")
    argparser.add_argument('--reset', action="store_true")
    argparser.add_argument('--test', action="store_true")
    argparser.add_argument("--iid", action="store_true")
    argparser.add_argument("--rln", type=int, default=6)
    argparser.add_argument("--neuromodulation", action="store_true")
    argparser.add_argument("--bias", type=float, default=0.0)
    argparser.add_argument("--save_name", type=str, default="performance")
    argparser.add_argument('--imagenet_path', help='Name of experiment', default="/users/s/b/sbeaulie/meta-learning_neuromodulation_for_catastrophic_forgetting/anml/imagenet_data/")

    args = argparser.parse_args()
    args.name = "/".join([args.dataset, "eval", str(args.epochs).replace(".", "_"), args.name])
    main(args)
