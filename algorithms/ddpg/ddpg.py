import numpy as np
from algorithms.ddpg.noise import OrnsteinUhlenbeckActionNoise
from algorithms.ddpg.replay_buffer import ReplayBuffer, Transition
from tensorboardX import SummaryWriter
import os
from utils.helper import save_checkpoint, np2tensor
from utils.runningmeanstd import RunningMeanStd
import torch.nn as nn
import torch
from torch.autograd import Variable
# from torchviz import make_dot
import pdb


def train(actor, target_actor, critic, target_critic,
          actor_optimizer, critic_optimizer, args):
    """Train the DDPG learner."""

    env = args['env']
    device = args['device']
    total_steps = args['total_steps']
    total_episodes = args['total_episodes']
    action_dim = env.action_space.shape[0]
    replay = ReplayBuffer(max_size=1000000)
    reward_buffer = np.zeros(args['log_reward_buffer_size'])
    reward_buffer_pos = 0
    # return_rms = RunningMeanStd(dim=1, device=device)
    state_rms = RunningMeanStd(dim=env.observation_space.shape[0])
    loss_fn = nn.MSELoss()

    if 'state_info' in args:
        state_rms.set_state(args['state_info'] + [args['total_steps']])

    if 'reward_info' in args:
        return_rms.set_state(args['reward_info'] + [args['total_steps']])

    checkpoint_dir = os.path.join('checkpoints', 'ddpg', args['env_name'])
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Summary writer for tensorboardX
    writer = {}
    writer['writer'] = SummaryWriter()

    s = env.reset()
    s = np2tensor(s, device).unsqueeze(0)
    # s = state_rms.normalize(s)
    done = False
    ep_reward = 0

    action_noise = OrnsteinUhlenbeckActionNoise(action_dim=action_dim)
    action_noise.scale = ((args['noise_scale'] - args['final_noise_scale']) *
                          max(0, args['exploration_end'] - total_episodes) /
                          args['exploration_end'] + args['final_noise_scale'])
    action_noise.reset()

    while total_steps < args['n_training_steps']:

        # while True:
        ou_noise = np2tensor(action_noise(), device)
        a = actor(s) + ou_noise
        a = a.clamp(-1, 1).detach()
        # a = actor(np2tensor(s, device))
        # a = a.data.to('cpu').numpy()
        next_s, r, done, _ = env.step(a.to('cpu').numpy()[0])
        # next_s = state_rms.normalize(next_s)
        # r = return_rms.normalize(r)
        ep_reward += r
        r = torch.Tensor([r]).to(device)
        done = torch.Tensor([int(done)]).to(device)
        next_s = np2tensor(next_s, device).unsqueeze(0)
        replay.update(s, a, r, next_s, done)
        s = next_s
        total_steps += 1

        if len(replay) <= args['batch_size'] and not done:
            continue

        # pdb.set_trace()

        for i in range(args['updates_per_step']):
            actor_loss = 0
            critic_loss = 0
            batch = replay.sample(size=args['batch_size'])
            batch = Transition(*zip(*batch))

            states = Variable(torch.cat(batch.state))
            actions = Variable(torch.cat(batch.action))
            rewards = Variable(torch.cat(batch.reward)).unsqueeze(1)
            terminals = Variable(torch.cat(batch.done)).unsqueeze(1)
            next_states = Variable(torch.cat(batch.next_state))

            target_next_a = target_actor(next_states)
            # target_next_q = return_rms.denormalize(target_critic(exp_next_s,
            #                                                      target_next_a)
            #                                        ).detach()
            target_next_q = target_critic(next_states, target_next_a)
            target_q = rewards + (1 - terminals) * args['gamma'] * target_next_q

            critic_optimizer.zero_grad()
            pred_q = critic(states, actions)
            critic_loss = loss_fn(pred_q, target_q)
            # for (n, p) in critic.named_parameters():
            #     if 'final' not in n:
            #         critic_loss += args['l2_reg'] * torch.norm(p.flatten(), p=2)
            critic_loss.backward()
            critic_optimizer.step()

            # target_q = return_rms.normalize(target_q)
            # actor_loss -= return_rms.denormalize(critic(exp_s, actor(exp_s)))
            actor_optimizer.zero_grad()
            actor_loss = -critic(states, actor(states))
            actor_loss = actor_loss.mean()
            # for (n, p) in actor.named_parameters():
            #     if 'final' not in n:
            #         actor_loss += args['l2_reg'] * torch.norm(p.flatten(), p=2)
            actor_loss.backward()
            actor_optimizer.step()

            target_actor.set_weighted_weights(actor, args['tau'])
            target_critic.set_weighted_weights(critic, args['tau'])

            writer['writer'].add_scalar('losses/actor_loss',
                                        actor_loss,
                                        (total_steps - 1) * args['updates_per_step'] + i)
            writer['writer'].add_scalar('losses/critic_loss',
                                        critic_loss,
                                        (total_steps - 1) * args['updates_per_step'] + i)

        if done:
            reward_buffer[reward_buffer_pos] = ep_reward
            reward_buffer_pos = ((reward_buffer_pos + 1) %
                                 args['log_reward_buffer_size'])
            writer['writer'].add_scalar('rewards/train_reward',
                                        ep_reward,
                                        total_episodes)
            writer['writer'].add_scalar('rewards/train_reward_buffer',
                                        reward_buffer.mean(),
                                        total_episodes)
            # writer['writer'].add_scalar('data/ep_len',
            #                             ep_len,
            #                             total_steps)
            if total_episodes % 10 == 0:
                print('Training Episode %d: Reward: %f '
                      'Avg. reward (last %d): %f' %
                      (total_episodes, ep_reward,
                       args['log_reward_buffer_size'], reward_buffer.mean()))
                test(actor, 1, args)
            total_episodes += 1
            s = env.reset()
            s = np2tensor(s, device).unsqueeze(0)
            # s = state_rms.normalize(s)
            ep_reward = 0
            action_noise.scale = ((args['noise_scale'] -
                                   args['final_noise_scale']) *
                                  max(0, args['exploration_end'] -
                                  total_episodes) / args['exploration_end'] +
                                  args['final_noise_scale'])
            action_noise.reset()

        save_checkpoint({'actor': actor.state_dict(),
                         'target_actor': target_actor.state_dict(),
                         'critic': critic.state_dict(),
                         'target_critic': target_critic.state_dict(),
                         'actor_optimizer': actor_optimizer.state_dict(),
                         'critic_optimizer': critic_optimizer.state_dict(),
                         'state_info': [state_rms.mean, state_rms.var],
                         # 'reward_info': [return_rms.mean, return_rms.var],
                         'total_steps': total_steps,
                         'total_episodes': total_episodes
                         },
                        filename=os.path.join(checkpoint_dir, '%d.pth.tar' %
                                              (total_episodes)))

    trained_vars = {'actor': actor, 'critic': critic}
    return trained_vars


def test(actor, n_test_episodes, args):
    env = args['env']

    for ep in range(n_test_episodes):
        s = env.reset()
        ep_reward = 0
        done = False

        while not done:
            s = np2tensor(s, args['device']).unsqueeze(0)
            a = actor(s).to('cpu').data.numpy()[0]
            next_s, r, done, _ = env.step(a)
            ep_reward += r
            s = next_s
            if args['visualize']:
                env.render()

        # print('Episode %d reward: %f' % (ep + 1, ep_reward))
        print('Test reward:', ep_reward)
