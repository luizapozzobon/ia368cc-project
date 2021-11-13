import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from wrapper import AtariARIWrapper
from itertools import chain
from probe import ProbeTrainer

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="BreakoutNoFrameskip-v4",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')

    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=8,
        help='the number of parallel game environments')
    parser.add_argument('--num-steps', type=int, default=128,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=4,
        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggles advantages normalization")
    parser.add_argument('--clip-coef', type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--ent-coef', type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
        help='the target KL divergence threshold')
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(gym_id, seed, idx, capture_video, run_name, atariari=True):
    def thunk():
        env = gym.make(gym_id)
        if atariari:
            env = AtariARIWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        # env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class Flatten(nn.Module):
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/
    def forward(self, x):
        return x.view(x.size(0), -1)


def init(module, weight_init, bias_init, gain=1):
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class NatureCNN(nn.Module):
    def __init__(self, input_channels, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        self.input_channels = input_channels
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        self.flatten = Flatten()

        self.final_conv_size = 64 * 9 * 6
        self.final_conv_shape = (64, 9, 6)
        self.main = nn.Sequential(
            init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 128, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(128, 64, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(self.final_conv_size, self.feature_size)),
            # nn.ReLU()
        )
        self.train()

    @property
    def local_layer_depth(self):
        return self.main[4].out_channels

    def forward(self, inputs, fmaps=False):
        f5 = self.main[:6](inputs)
        f7 = self.main[6:8](f5)
        out = self.main[8:](f7)
        if fmaps:
            return {
                "f5": f5.permute(0, 2, 3, 1),
                "f7": f7.permute(0, 2, 3, 1),
                "out": out,
            }
        return out


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()

        self.feature_size = 256

        self.network = NatureCNN(input_channels=4, feature_size=self.feature_size)

        # self.network = nn.Sequential(
        #     layer_init(nn.Conv2d(4, 32, 8, stride=4)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(64, 64, 3, stride=1)),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     layer_init(nn.Linear(64 * 7 * 7, 512)),
        #     nn.ReLU(),
        # )

        self.actor = layer_init(
            nn.Linear(self.feature_size, envs.single_action_space.n), std=0.01
        )
        self.critic = layer_init(nn.Linear(self.feature_size, 1), std=1)

        # TODO should we add the Classifier class (with Bilinear layer?)
        self.classifier1 = nn.Linear(
            self.feature_size, self.network.local_layer_depth
        ).to(
            device
        )  # x1 = global, x2=patch, n_channels = 32
        self.classifier2 = nn.Linear(
            self.network.local_layer_depth, self.network.local_layer_depth
        ).to(device)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)  # self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def get_representation(self, x, fmaps=True):
        return self.network(x / 255.0, fmaps=fmaps)

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    episode_labels = [[] for _ in range(args.num_envs)]

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(
                        f"global_step={global_step}, episodic_return={item['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", item["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", item["episode"]["l"], global_step
                    )
                    break


            for i, info_i in enumerate(info):
                if(isinstance(episode_labels[i], dict)):
                  pass
                else: episode_labels[i].append([info_i["labels"]])
                #if done[i] != 1:
                #    if "labels" in info_i.keys():
                #        episode_labels[i][-1].append(info_i["labels"])
                #else:
                #    if "labels" in info_i.keys():
                #        episode_labels[i].append([info_i["labels"]])


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizaing the policy and value network
        indexes = np.arange(args.batch_size)
        indexes_matrix = indexes.reshape(obs.shape[:2])
        b_inds = indexes.copy()
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                # Constrastive learning - mila-iqia stdim.py
                x_t_ind = mb_inds[mb_inds >= 128]
                x_t_prev_ind = x_t_ind - 128
                x_t, x_t_prev = b_obs[x_t_ind], b_obs[x_t_prev_ind]

                f_t_maps = agent.get_representation(x_t)
                f_t_prev_maps = agent.get_representation(x_t_prev)

                f_t = f_t_maps["out"]  # N x feature_size
                f_t_prev = f_t_prev_maps["f5"]  # N x 11 x 8 x 128
                sy = f_t_prev.size(1)  # 11
                sx = f_t_prev.size(2)  # 8

                N = f_t.size(0)
                loss1 = 0.0
                predictions = agent.classifier1(f_t)  # N x 128
                for y in range(sy):
                    for x in range(sx):
                        positive = f_t_prev[:, y, x, :]  # N x 128
                        logits = torch.matmul(predictions, positive.t())  # N x N
                        step_loss = F.cross_entropy(logits, torch.arange(N).to(device))
                        loss1 += step_loss
                loss1 = loss1 / (sx * sy)

                # Loss 2: f5 patches at time t, with f5 patches at time t-1
                f_t = f_t_maps["f5"]
                loss2 = 0.0
                for y in range(sy):
                    for x in range(sx):
                        predictions = agent.classifier2(f_t[:, y, x, :])
                        positive = f_t_prev[:, y, x, :]
                        logits = torch.matmul(predictions, positive.t())
                        step_loss = F.cross_entropy(logits, torch.arange(N).to(device))
                        loss2 += step_loss
                loss2 = loss2 / (sx * sy)
                contrastive_loss = loss1 + loss2
                # End of contrastive learning

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                contr_coef = 0.1

                ## TODO: + or - contrastive_loss?
                ## TODO: contr_coef = ?
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                    + contrastive_loss * contr_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Testing probe
                #probes = {k: LinearProbe(input_dim=256, num_classes=256).to(device) for k in positive.t().keys()}
                #f = mb_inds.detach()
                #preds = [probe(f) for probe in probes]
                ##preds = [pred.cpu().detach().numpy() for pred in preds]
                #preds = [np.argmax(pred, axis=1) for pred in preds]
                #print(preds)

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        #print(f"shape obs {obs.shape}")
        #print(f"shape b_obs {b_obs.shape}")

        episode_labels_chained = list(chain.from_iterable(episode_labels)) 
        arr_episode_labels = np.array(episode_labels_chained)
        #print(f"shape labels after chain: {arr_episode_labels.shape}")
        x_chop = int(len(episode_labels_chained)/b_obs.shape[0])
        #print(f"x_chop: {x_chop}")

        chop_labels = episode_labels_chained[(x_chop-1)*b_obs.shape[0]:]
        inds = np.arange(len(chop_labels))
        split_ind = int(0.8 * len(inds))
        val_split_ind, te_split_ind = int(0.7 * len(inds)), int(0.8 * len(inds))
        assert val_split_ind > 0 and te_split_ind > val_split_ind,\
            "Not enough episodes to split into train, val and test. You must specify more steps"
        tr_labels, val_labels, test_labels = chop_labels[:val_split_ind], \
                                             chop_labels[val_split_ind:te_split_ind], chop_labels[te_split_ind:]
        #print(f"{len(tr_labels)}, {len(val_labels)}, {len(test_labels)}")
        b_obs_tr, b_obs_val, b_obs_tes =  b_obs[:val_split_ind], \
                                          b_obs[val_split_ind:te_split_ind], b_obs[te_split_ind:]
        #print(f"{len(b_obs_tr)}, {len(b_obs_val)}, {len(b_obs_tes)}")
        
        trainer1 = ProbeTrainer(encoder=agent.network,
                               epochs=args.update_epochs,
                               batch_size=args.batch_size)

        trainer1.train(b_obs_tr, b_obs_val, tr_labels, val_labels)
        test_acc1, test_f1score1 = trainer1.test(b_obs_tes, test_labels)

        #trainer2 = ProbeTrainer(encoder=agent.classifier2,
        #                       epochs=args.update_epochs,
        #                       batch_size=args.batch_size)#

        #trainer2.train(b_obs_tr, b_obs_val, tr_labels, val_labels)
        #test_acc2, test_f1score2 = trainer2.test(b_obs_tes, test_labels)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "losses/contrastive_loss", contrastive_loss.item(), global_step
        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        for key in test_acc1:
          writer.add_scalar(
              "probe1_test_acc_"+key, test_acc1[key], global_step
          )
        for key in test_f1score1:
          writer.add_scalar(
              "probe1_test_f1score_"+key, test_f1score1[key], global_step
          )
        #writer.add_scalar(
        #    "probe2_test_acc", test_acc2, global_step
        #)
        #writer.add_scalar(
        #    "probe2_test_f1score", test_f1score2, global_step
        #)
    envs.close()
    writer.close()
