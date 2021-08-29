import torch
import logging
import argparse

import numpy as np
import utils.utils as utils
import model.modelfactory as mf
import datasets.task_sampler as ts
import datasets.datasetfactory as df

from tensorboardX import SummaryWriter
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification

logger = logging.getLogger('experiment')

def main(args):

    print("Launching meta-learning protocol...")

    utils.set_seed(args.seed*int(args.pass_num))
        
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    args.classes = list(range(args.num_classes))

    dataset = \
            df.DatasetFactory.get_dataset(args.dataset, background=True, train=True, all=True)
    dataset_test = \
            df.DatasetFactory.get_dataset(args.dataset, background=True, train=False, all=True)

    # Data iterators used for meta-learning
    iterator_test = torch.utils.data.DataLoader(dataset_test, batch_size=5,
                                                shuffle=True, num_workers=1)

    iterator_train = torch.utils.data.DataLoader(dataset, batch_size=5,
                                                 shuffle=True, num_workers=1)

    sampler = ts.SamplerFactory.get_sampler(args.dataset, args.classes, dataset, dataset_test)

    # Create model:

    config = mf.ModelFactory.get_model(args.dataset,
                                       channels=args.channels,
                                       nm_channels=args.nm_channels,
                                       nm_rep_size=args.nm_rep_size,
                                       rep_size=args.rep_size,
                                       treatment=args.treatment)

    maml = MetaLearingClassification(args,
                                     config,
                                     bias=args.bias,
                                     channels=args.channels,
                                     nm_channels=args.nm_channels,
                                     rep_size=args.rep_size,
                                     nm_rep_size=args.nm_rep_size,
                                     device=device)

    maml = maml.to(device)
    utils.freeze_layers(args.rln, maml)
    
    # Begin meta-learning:

    counter = 0
    for step in range(args.steps):

        inner_loop_task = np.random.choice(args.classes, args.tasks, replace=False)
        inner_iterators = []
        for t in inner_loop_task:
            inner_iterators.append(sampler.sample_task([t]))

        rand_outer_iterator = sampler.get_complete_iterator()
        x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(inner_iterators, 
                                                               rand_outer_iterator,
                                                               steps=args.num_inner_steps, 
                                                               reset=not args.no_reset)

        if torch.cuda.is_available():
            x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

        accuracy, loss = maml(x_spt, y_spt, x_qry, y_qry)

        if step % 100 == 0:
            torch.save(maml.net, "{0}_Bias={1}_Seed={2}".format(args.model_name, 
                                                                args.bias, 
                                                                args.seed))   

        print('Iteration={0} \t  Accuracy={1}'.format(int(step), accuracy))

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--treatment', type=str, help='ANML or TSAR', default="TSAR")
    argparser.add_argument('--steps', type=int, help='epoch number', default=25000)
    argparser.add_argument('--pass_num', type=float, help='Daisy chain number', default=1)
    argparser.add_argument('--model_name', help='Name of model to be saved', default='mymodel.net')
    argparser.add_argument('--inner_gradients', type=str, help='first or second order')
    argparser.add_argument('--checkpoint', help='Use a checkpoint model', action='store_true')
    argparser.add_argument('--saved_model', help='Saved model to load', default='my_model.net')
    argparser.add_argument('--seed', type=int, help='Seed for random', default=10000)
    argparser.add_argument('--seeds', type=int, nargs='+', help='n way', default=[10])
    argparser.add_argument('--tasks', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='outer learning rate', default=1e-2)
    argparser.add_argument('--update_lr', type=float, help='inner learning rate', default=0.01)
    argparser.add_argument('--num_inner_steps', type=int, help='inner update steps', default=20)
    argparser.add_argument('--name', help='Name of experiment', default="mrcl_classification")
    argparser.add_argument('--dataset', help='Name of experiment', default="omniglot")
    argparser.add_argument("--commit", action="store_true")
    argparser.add_argument("--no-reset", action="store_true")
    argparser.add_argument("--rln", type=int, default=10)
    argparser.add_argument('--model', type=str, help='epoch number', default="none")
    argparser.add_argument("--bias", type=float, default=-8)
    argparser.add_argument('--num_classes', type=int, help='# unique classes', default=963)
    argparser.add_argument("--channels", type=int, default=112)
    argparser.add_argument("--nm_channels", type=int, default=196)
    argparser.add_argument("--nm_rep_size", type=int, default=1728, help="NM representation")
    argparser.add_argument("--rep_size", type=int, default=112, help="Classifier representation")

    args = argparser.parse_args()
    args.data_path = "../data/omni"

    args.name = "/".join([args.dataset, str(args.meta_lr).replace(".", "_"), args.name])
    print(args)
    
    main(args)
