import copy
import torch
import logging

import numpy as np
import model.learner as Learner

from torch import nn
from torch import optim
from torch.nn import functional as F

logger = logging.getLogger("experiment")

class MetaLearingClassification(nn.Module):

    def __init__(self, args, config, bias=-8, channels=72, nm_channels=196, rep_size=72, nm_rep_size=1764, device='cuda'):

        super(MetaLearingClassification, self).__init__()

        self.bias = bias
        self.device = device
        self.meta_iteration = 0
        self.meta_lr = args.meta_lr
        self.update_lr = args.update_lr
        self.treatment = args.treatment
        self.inner_gradients = args.inner_gradients
        self.num_inner_steps = args.num_inner_steps

        if self.inner_gradients == 'first_order':
            print("Computing first-order gradients")
        else:
            print("Computing second-order gradients")

        self.net = Learner.Learner(config, 
                                   bias, 
                                   channels, 
                                   nm_channels, 
                                   rep_size, 
                                   nm_rep_size,
                                   device,
                                   args.treatment)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.layers_to_fix = []

    def reset_classifer(self, class_to_reset):
        """
        Code for resetting weights in CP that innervate the given
        class node. This prevents overfitting during meta-learning

        """

        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))

    def sample_training_data(self, iterators, it2, steps=2, reset=True):

        """
        Code for sampling images from the supplied iterators.

        """

        # Sample data for inner and meta updates

        x_traj = []
        y_traj = []
        x_rand = []
        y_rand = []

        counter = 0

        class_cur = 0
        class_to_reset = 0
        for it1 in iterators:
            for img, data in it1:
                class_to_reset = data[0].item()
                if reset:
                    # Resetting weights corresponding to classes in the inner updates; 
                    # this prevents the learner from memorizing the data 
                    # (which would kill the gradients due to inner updates)
                    self.reset_classifer(class_to_reset)

                #self.net.cuda()
                counter += 1
                x_traj.append(img)
                y_traj.append(data)
                if counter % int(steps / len(iterators)) == 0:
                    class_cur += 1
                    break

        # To handle a corner case; nothing interesting happening here
        if len(x_traj) < steps:
            it1 = iterators[-1]
            for img, data in it1:
                counter += 1
                x_traj.append(img)
                y_traj.append(data)
                if counter % int(steps % len(iterators)) == 0:
                    break

        # Sampling the random batch of data
        counter = 0
        for img, data in it2:
            if counter == 1:
                break
            x_rand.append(img)
            y_rand.append(data)
            counter += 1

        class_cur = 0
        counter = 0
        x_rand_temp = []
        y_rand_temp = []
        for it1 in iterators:
            for img, data in it1:
                counter += 1
                x_rand_temp.append(img)
                y_rand_temp.append(data)
                if counter % int(steps / len(iterators)) == 0:
                    class_cur += 1
                    break

        y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
        x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)
        x_traj = torch.stack(x_traj) 
        y_traj = torch.stack(y_traj) 
        x_rand = torch.stack(x_rand) 
        y_rand = torch.stack(y_rand)

        x_rand = torch.cat([x_rand, x_rand_temp], 1)
        y_rand = torch.cat([y_rand, y_rand_temp], 1)

        return(x_traj, y_traj, x_rand, y_rand)


    def inner_update(self, x, fast_weights, y, bn_training):

        if self.inner_gradients == 'first_order':

            logits = self.net(x, fast_weights, bn_training=bn_training)
            loss = F.cross_entropy(logits, y)

            if fast_weights is None:
                fast_weights = self.net.parameters()

            grad = torch.autograd.grad(loss, fast_weights, allow_unused=False)
            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

            for params_old, params_new in zip(self.net.parameters(), fast_weights):
                params_new.learn = params_old.learn

            return(fast_weights)

        elif self.inner_gradients == 'second_order':

            logits = self.net(x, fast_weights, bn_training=bn_training)
            loss = F.cross_entropy(logits, y)

            inner_params = list(range(20, 34))

            if fast_weights is None:
                fast_weights = self.net.parameters()

            grad = torch.autograd.grad(loss, 
                                       [fast_weights[w] for w in inner_params],
                                       allow_unused=False, 
                                       create_graph=True)

            new_weights = []
            counter = 0
            for idx in range(len(fast_weights)):

                if idx in inner_params:
                    new_weight = fast_weights[idx] - self.update_lr*grad[counter]
                    new_weight.learn = fast_weights[idx].learn
                    counter += 1

                else:
                    new_weight = fast_weights[idx]
                    new_weight.learn = fast_weights[idx].learn

                new_weights.append(new_weight)

            return(new_weights)

    def meta_loss(self, x, fast_weights, y, bn_training):
        """
        Code for computing outer loop loss.

        """

        logits = self.net(x, 
                          fast_weights, 
                          bn_training=bn_training, 
                          outer_loop=True, 
                          analysis=False)

        loss_q = F.cross_entropy(logits, y)

        return(loss_q, logits)

    def eval_accuracy(self, logits, y):

        pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item()
        return(correct)

    def forward(self, x_traj, y_traj, x_rand, y_rand):

        """
        Forward propagation of inputs sampled from the supplied iterators
        covering the inner and outer meta-learning loops.

        """

        fast_weights = self.inner_update(x_traj[0], None, y_traj[0], False)

        for k in range(self.num_inner_steps):

            fast_weights = self.inner_update(x_traj[k], 
                                             fast_weights=fast_weights, 
                                             y=y_traj[k], 
                                             bn_training=False)

        meta_loss, logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], bn_training=False)
      
        with torch.no_grad():
            pred_q = F.softmax(logits, dim=1).argmax(dim=1)
            classification_accuracy = torch.eq(pred_q, y_rand[0]).sum().item()  # convert to numpy

        self.net.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        classification_accuracy /= len(x_rand[0])
        self.meta_iteration += 1

        return(classification_accuracy, meta_loss)

def main():
    pass

if __name__ == '__main__':
    main()
