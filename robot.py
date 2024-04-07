#!/usr/bin/env python3

import argparse
import pickle
from collections import namedtuple
import copy

import os, time, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from src.turtlebot3_dqn.environment_ampg import Env
import rospy
from geometry_msgs.msg import Twist,Point,Quaternion


Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'dw'])

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_size, 100)
        self.mean_layer = nn.Linear(100, args.action_size)
        self.log_std = nn.Parameter(
            torch.zeros(1, args.action_size))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Actor(nn.Module):  # discrete action set
    def __init__(self, num_state, num_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_state):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

        if args.use_orthogonal_init and args.continu:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.state_value)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class ReinforceAgent():
    def __init__(self, isdirection, args):
        self.isdirection = isdirection
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        if isdirection:
            self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/ampg/direction')
        else:
            self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/ampg/speed')
        self.load_model = args.load_model
        self.state_size = args.state_size
        self.action_size = args.action_size
        self.episode_step = 6000
        # self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.batch_size = 64
        self.memory = []
        self.ppo_update_time = 10
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.training_step = 0
        self.counter = 0
        self.action_loss = -1
        self.value_loss = -1
        # self.weight_reg = args.IPM_weight  # IPM uncertainty set 1e-5/1e-4
        if isdirection:
            self.model_path = args.model_path_d
        else:
            self.model_path = args.model_path_s
        self.continu = args.continu
        self.max_action = args.max_action
        self.use_adv_norm = args.use_adv_norm
        self.noise = args.noise
        self.lamda = args.lamda  # GAE parameter for variance reduction
        self.td = args.td

        if self.continu:
            self.actor = Actor_Gaussian(args)
        else:
            self.actor = Actor(self.state_size, self.action_size)
        self.critic = Critic(self.state_size)
        if isdirection:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), 1e-4)
            self.critic_net_optimizer = optim.Adam(self.critic.parameters(), 3e-4)
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), 1e-5)
            self.critic_net_optimizer = optim.Adam(self.critic.parameters(), 3e-4)
        self.eta = 0
        self.b = 0
        self.alpha = args.alpha
        self.nu = args.nu

        # if self.load_model:
        #     load_model1 = torch.load(self.model_path)
        #     self.actor.load_state_dict(load_model1['actor_net'])
        #     self.critic.load_state_dict(load_model1['critic_net'])
        #     print("load model:", self.model_path)

    def select_action(self, state):  # return action and action_prob/ action_log_prob (for discrete/continuous)
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            if self.continu:
                dist = self.actor.get_dist(state)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
                return a.numpy().flatten(), a_logprob.numpy().flatten()
            else:
                action_prob = self.actor(state)
                c = Categorical(action_prob)
                action = c.sample()
                return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic(state)
        return value.item()

    def save_param(self, e):
        state = {'actor_net': self.actor.state_dict(), 'critic_net': self.critic.state_dict(),
                 'actor_optimizer': self.actor_optimizer.state_dict(), 'critic_optimizer': self.critic_net_optimizer,
                 'epoch': e}
        if self.isdirection:
            torch.save(state, args.save_model_path_d + str(e) + "b.pt")
        else:
            torch.save(state, args.save_model_path_s + str(e) + "b.pt")

    def store_transition(self, transition):
        self.memory.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.memory], dtype=torch.float)
        next_state = torch.tensor([t.next_state for t in self.memory], dtype=torch.float)
        dw = torch.tensor([t.dw for t in self.memory], dtype=torch.float).view(-1, 1)  # done or not
        if self.continu:
            action = torch.tensor([t.action for t in self.memory], dtype=torch.float).view(-1, 1)
        else:
            action = torch.tensor([t.action for t in self.memory], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.memory]
        old_action_prob = torch.tensor([t.a_log_prob for t in self.memory], dtype=torch.float).view(-1, 1)

        with torch.no_grad():  # calculate target V/Q
            if not self.td:
                R = 0
                Gt = []
                for r in reward[::-1]:
                    R = r + R
                    Gt.insert(0, R)
                # print(type(Gt), Gt)
                # sys.exit()
                target = torch.tensor(Gt, dtype=torch.float).view(-1,1)
                V = self.critic(state)
                adv = target - V

            else:
                vs = self.critic(state)
                vs_ = self.critic(next_state)
                R, avgV = 0, 0
                for r in reward:
                    R += r
                R = R/len(reward)
                for val in vs.view(-1):
                    avgV += val.item()
                avgV = avgV/vs.shape[0]

                self.eta = (1-self.alpha) * self.eta + self.alpha * R
                self.b = (1-self.alpha) * self.b + self.alpha * avgV
                adv, gae = [], 0
                reward = torch.tensor(reward, dtype=torch.float).view(-1, 1)

                deltas = reward - torch.ones(reward.shape) * self.eta + (1.0 - dw) * vs_ - vs
                for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(dw.flatten().numpy())):
                    gae = delta + self.lamda * gae * (1.0 - d)
                    adv.insert(0, gae)
                adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
                target = reward - torch.ones(reward.shape) * (self.eta + self.nu * self.b) + vs

        print('agent update')
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), self.batch_size, False):
                if self.training_step%1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_ep, self.training_step))

                advantage = adv[index]
                if self.use_adv_norm:
                    advantage = ((advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-5))

                if self.continu:
                    dist_now = self.actor.get_dist(state[index])
                    a_log_prob_now = dist_now.log_prob(action[index])
                    ratio = torch.exp(a_log_prob_now.sum(1, keepdim=True) - old_action_prob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)
                else:
                    action_prob = torch.gather(self.actor(state[index]), 1, action[index].view(-1, 1))
                    ratio = (action_prob / old_action_prob[index])

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                V_s = self.critic(state[index])

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # max obj = min -obj
                self.action_loss = torch.max(action_loss)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(target[index], V_s)
                self.value_loss = torch.max(value_loss)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.memory[:]  # clear experience


def train(args):
    agentD = ReinforceAgent(True, args)
    agentS = ReinforceAgent(False, args)
    start_time = time.time()

    if args.load_model:
        load_modelD = torch.load(args.model_path + "/direction/seed" + str(args.seed) + "/" + str(args.iter) + "b.pt")
        agentD.actor.load_state_dict(load_modelD['actor_net'])
        agentD.critic.load_state_dict(load_modelD['critic_net'])
        load_modelS = torch.load(args.model_path + "/speed/seed" + str(args.seed) + "/" + str(args.iter) + "b.pt")
        agentS.actor.load_state_dict(load_modelS['actor_net'])
        agentS.critic.load_state_dict(load_modelS['critic_net'])
        print('success load models from ',args.model_path + "/direction/seed" + str(args.seed) + "/" + str(args.iter) + "b.pt")

    # for evaluation
    best_speed = -1
    best_actorD, best_actorS = None, None
    best_epoch = -1

    record_speed = []
    record_avg = []

    for e in range(491,1500):
        print('reset! iteration ', e)
        # last_time = time.time()
        R, speed = 0, 0
        state = env.reset(e)
        episode_reward_sum = 0
        episode_step = 600000

        for t in range(episode_step):
            actionD, action_probD = agentD.select_action(state)
            actionS, action_probS = agentS.select_action(state)
            if (e<100 and np.random.uniform(0,1)<0.1):
                # print(action, type(action))
                # sys.exit()
                actionD[0] = np.random.uniform(0, args.max_action)
                # actionS[0] = np.random.uniform(0, args.max_speed)
            next_state, reward, done, _ = env.step(actionD, actionS, e)
            R += reward

            transD = Transition(state, actionD, action_probD, reward, next_state, done)
            transS = Transition(state, actionS, action_probS, reward, next_state, done)
            agentD.store_transition(transD)
            agentS.store_transition(transS)
            state = next_state
            episode_reward_sum += reward
            if e % 10 == 0 or (e%5==0 and e<30):
                agentD.save_param(e)
                agentS.save_param(e)
            if e % 100 == 0:
                if best_actorD:
                    temp = {'actor_net': best_actorD.state_dict(), 'epoch': best_epoch, 'speed': best_speed}
                    torch.save(temp,
                               args.save_model_path_d + "best_actor_" + "speed_" + str(
                                   round(best_speed, 5)) + "_epoch" + str(best_epoch) + ".pt")
                    temp = {'actor_net': best_actorS.state_dict(), 'epoch': best_epoch, 'speed': best_speed}
                    torch.save(temp,
                               args.save_model_path_s + "best_actor_" + "speed_" + str(
                                   round(best_speed, 5)) + "_epoch" + str(best_epoch) + ".pt")
            if t >= 1500:
                rospy.loginfo("time out!")
                done = True
                # env.goal_x, env.goal_y = env.respawn_goal.getPosition(e, True, delete=True)
                # env.goal_distance = env.getGoalDistace()
                # print('Goal Distance : ', env.goal_distance)
            if done or env.get_goalbox:
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                # cur_time = time.time()
                if t>1:
                    speed = (env.goal_distance / t)
                # if (args.noise and agent.counter >= 64) or not args.noise:  # update only when #sample>64
                if agentD.counter<=1 or agentS.counter<=1:
                    del agentD.memory[:]
                    agentD.counter = 0
                    del agentS.memory[:]
                    agentS.counter = 0
                    break
                agentD.update(e)
                agentS.update(e)
                rospy.loginfo('Ep: %d score: %.2f memory: %d episode_step: %.2f time: %d:%02d:%02d', e,
                              episode_reward_sum, agentD.counter, t, h, m, s)
                agentD.counter = 0
                agentS.counter = 0

                env.goal_x, env.goal_y = env.respawn_goal.getPosition(e, True, delete=True)
                env.goal_distance = env.getGoalDistace()
                print('Goal Distance : ', env.goal_distance)

                if env.get_goalbox and e > 80:
                    # speed = (env.goal_distance/(cur_time-last_time))
                    best_epoch = e
                    if best_speed < 0 or speed > best_speed:
                        best_speed = speed
                        best_actorD = copy.deepcopy(agentD.actor)
                        best_actorS = copy.deepcopy(agentS.actor)
                        print('update best model, speed:', speed)
                if not env.get_goalbox:
                    speed = 0
                env.get_goalbox = False
                break
        if t>1:
            record_avg.append(R/t)
            record_speed.append(speed)
            print('avg, speed:', R/t, speed)

    if best_actorD:
        temp = {'actor_net': best_actorD.state_dict(), 'epoch': best_epoch, 'speed': best_speed}
        torch.save(temp,
                   args.save_model_path_d + "best_actor_" + "speed_" + str(
                       round(best_speed, 5)) + "_epoch" + str(best_epoch) + ".pt")
        temp = {'actor_net': best_actorS.state_dict(), 'epoch': best_epoch, 'speed': best_speed}
        torch.save(temp,
                   args.save_model_path_s + "best_actor_" + "speed_" + str(
                       round(best_speed, 5)) + "_epoch" + str(best_epoch) + ".pt")


def eval(args):
    # load_modelD = torch.load(args.model_path+"/direction/seed1/"+str(args.eval_iter)+"b.pt")
    # load_modelS = torch.load(args.model_path + "/speed/seed1/" + str(args.eval_iter) + "b.pt")
    record_avg_speed = []
    record_count = []
    record_iter = []
    # for iter in [i*5 for i in range(7)]+[(i+4)*10 for i in range(66)]:
    for iter in [i*10 for i in range(50,150)]:
        load_modelD = torch.load(args.model_path + "/direction/seed" + str(args.seed) + "/" + str(iter) + "b.pt")
        load_modelS = torch.load(args.model_path + "/speed/seed"+ str(args.seed) + "/" + str(iter) + "b.pt")
        actorD = Actor_Gaussian(args)
        actorD.load_state_dict(load_modelD['actor_net'])
        actorS = Actor_Gaussian(args)
        actorS.load_state_dict(load_modelS['actor_net'])
        actorD.eval()
        actorS.eval()
        print("load actor from:", args.model_path + "/direction/seed" + str(args.seed) + "/" + str(iter) + "b.pt")
        N = 20
        record_iter.append(iter)
        avgSpeed, count = 0, 0
        for i in range(N):
            print('seed ', i)
            np.random.seed(i)
            state = env.reset()
            env.goal_distance = env.getGoalDistace()
            # print('test1 ', env.goal_distance)
            # print('test:', env.goal_x, env.goal_y, env.goal_distance)
            # traj_x, traj_y = [env.position.x], [env.position.y]
            episode_reward_sum = 0
            episode_step = 6000

            for t in range(episode_step):
                state = torch.from_numpy(state).float().unsqueeze(0)

                distD = actorD.get_dist(state)
                aD = distD.sample()  # Sample the action according to the probability distribution
                aD = torch.clamp(aD, -args.max_action, args.max_action)
                actionD = aD.numpy().flatten()
                distS = actorS.get_dist(state)
                aS = distS.sample()  # Sample the action according to the probability distribution
                aS = torch.clamp(aS, -args.max_action, args.max_action)
                actionS = aS.numpy().flatten()

                next_state, reward, done, _ = env.step(actionD, actionS, -1)

                state = next_state
                episode_reward_sum += reward
                if t >= 500:
                    rospy.loginfo("time out!")
                    done = True
                if done or env.get_goalbox:
                    if t<=1:
                        break

                    if env.get_goalbox:
                        count +=1
                        avgSpeed += np.sqrt(env.goal_x**2+ env.goal_y**2)/t
                        print('speed ', np.sqrt(env.goal_x**2+ env.goal_y**2)/t)
                    env.pub_cmd_vel.publish(Twist())

                    env.goal_x, env.goal_y = env.respawn_goal.getPosition(300, True, delete=True)
                    env.get_goalbox = False
                    break
        if count:
            print('success times:', count, 'average speed:', avgSpeed/count)
            record_avg_speed.append(avgSpeed/count)
        else:
            print('success times: 0')
            record_avg_speed.append(0)
        record_count.append(count)

    with open(args.save_model_path_d+"record_part2", "wb") as fp:  # Pickling
        pickle.dump([record_count, record_iter, record_avg_speed], fp)


if __name__ == '__main__':
    #########HYPERPARAMETER###########
    parser = argparse.ArgumentParser("Hyperparameters")
    parser.add_argument("--state_size", type=int, default=2, help=" ")
    parser.add_argument("--action_size", type=int, default=1,
                        help="when discrete, action size is order. when continuous action size is dim")
    parser.add_argument("--max_action", type=float, default=2,
                        help="continuous action belongs to [-max_action,max_action] ")
    parser.add_argument("--max_speed", type=float, default=1,
                        help="continuous action belongs to [0,max_action] ")
    parser.add_argument("--continu", type=bool, default=True, help="F:discrete, T:continuous")
    parser.add_argument("--train", type=bool, default=False, help="F:evaluate, will automatically read from model_path, T:train")
    parser.add_argument("--load_model", type=bool, default=True, help="T: load model from path for TRAIN not TEST")
    parser.add_argument("--model_path", type=str,
                        default="catkin_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/save_model/ampg",
                        help="saved model path")
    parser.add_argument("--model_path_d", type=str,
                        default="catkin_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/save_model/ampg",
                        help="saved model path")
    parser.add_argument("--model_path_s", type=str,
                        default="catkin_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/save_model/ampg/speed/best_actor.pt",
                        help="saved model path")
    parser.add_argument("--save_model_path_d", type=str,
                        default="catkin_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/save_model/ampg/direction/seed/",
                        help="saving trained model path")
    parser.add_argument("--save_model_path_s", type=str,
                        default="catkin_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/save_model/ampg/speed/seed/",
                        help="saving trained model path")
    parser.add_argument("--goal_x", type=float, default=1, help=" ")
    parser.add_argument("--goal_y", type=float, default=-1, help=" ")
    parser.add_argument("--next_steps", type=int, default=2, help="Number of next states")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="advantage normalization")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="orthogonal initialization")
    parser.add_argument("--use_tanh", type=float, default=True, help="tanh activation function")
    parser.add_argument("--adaptive_alpha", type=float, default=False, help="adaptive entropy regularization")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--noise", type=bool, default=False, help="add noise into action of env step")
    parser.add_argument("--reset_noise_low", type=float, default=1)
    parser.add_argument("--reset_noise_high", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--td", type=bool, default=True, help="td loss or Monte Carlo for target")
    parser.add_argument("--dynamicSpeed", type=bool, default=False, help="use speed randomization as dynamic randomization")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="moving rate of average reward and ")
    parser.add_argument("--nu", type=float, default=0.1,
                        help="average value constraint lagrange cof")
    parser.add_argument("--iter", type=int, default=490, help="average value constraint lagrange cof")

    args, unknown = parser.parse_known_args()
    ##############END#################
    rospy.init_node('turtlebot3_ppo')
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('ampg seed', seed)
    if args.train:
        print('training')
    else:
        if args.noise:
            print('evaluating')
        else:
            print('evaluate without noise')
    env = Env(args)
    if args.train:
        print('train start')
        train(args)
    else:
        print('evaluate start')
        eval(args)

    print("end")
