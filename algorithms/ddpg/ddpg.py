import numpy as np
from algorithms.ddpg.noise import OrnsteinUhlenbeckActionNoise
from algorithms.ddpg.replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter
import os
from utils.helper import save_checkpoint, np2tensor
import torch.nn as nn
import torch


def train(actor, target_actor, critic, target_critic,
          actor_optimizer, critic_optimizer, args):
    """
    Training the DDPG learner: There are a few things to note:
    - Observations and returns are not normalized.
    - Layer normalization is used for the actor and critic networks.
    """

    env = args['env']
    device = args['device']
    total_steps = args['total_steps']
    total_episodes = args['total_episodes']
    action_dim = env.action_space.shape[0]
    replay = ReplayBuffer(max_size=1000000)
    reward_buffer = np.zeros(args['log_reward_buffer_size'])
    reward_buffer_pos = 0
    loss_fn = nn.MSELoss()

    checkpoint_dir = os.path.join('checkpoints', 'ddpg', args['env_name'])
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Summary writer for tensorboardX
    writer = {}
    writer['writer'] = SummaryWriter()

    s = env.reset()
    s = np2tensor(s, device).unsqueeze(0)
    done = False
    ep_reward = 0

    action_noise = OrnsteinUhlenbeckActionNoise(action_dim=action_dim)
    action_noise.scale = ((args['noise_scale'] - args['final_noise_scale']) *
                          max(0, args['exploration_end'] - total_episodes) /
                          args['exploration_end'] + args['final_noise_scale'])
    action_noise.reset()

    while total_steps < args['n_training_steps']:

        ou_noise = np2tensor(action_noise(), device)
        a = actor(s) + ou_noise
        a = a.clamp(-1, 1).detach()
        next_s, r, done, info = env.step(a.to('cpu').numpy()[0])
        ep_reward += r
        r = torch.Tensor([r]).to(device)
        done = torch.Tensor([int(done)]).to(device)
        next_s = np2tensor(next_s, device).unsqueeze(0)
        replay.update(s, a, r, next_s, done)
        s = next_s
        total_steps += 1

        if len(replay) <= args['batch_size'] and not done:
            continue

        for i in range(args['updates_per_step']):
            actor_loss = 0
            critic_loss = 0
            (states, actions, rewards,
             terminals, next_states) = replay.sample(size=args['batch_size'])

            target_next_a = target_actor(next_states)
            target_next_q = target_critic(next_states, target_next_a)
            target_q = rewards + (1 - terminals) * args['gamma'] * target_next_q

            critic_optimizer.zero_grad()
            pred_q = critic(states, actions)
            critic_loss = loss_fn(pred_q, target_q)
            critic_loss.backward()
            critic_optimizer.step()

            actor_optimizer.zero_grad()
            actor_loss = -critic(states, actor(states))
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            actor_optimizer.step()

            target_actor.soft_update(actor, args['tau'])
            target_critic.soft_update(critic, args['tau'])

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

            args['total_episodes'] = total_episodes

            if total_episodes % 10 == 0:
                print('Training Episode %d: Reward: %f '
                      'Avg. reward (last %d): %f' %
                      (total_episodes, ep_reward,
                       args['log_reward_buffer_size'], reward_buffer.mean()))
                test(actor, 1, args)

            total_episodes += 1

            s = env.reset()
            s = np2tensor(s, device).unsqueeze(0)
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
        steps = 0

        while not done:
            s = np2tensor(s, args['device']).unsqueeze(0)
            a = actor(s).to('cpu').data.numpy()[0]
            next_s, r, done, info = env.step(a)
            ep_reward += r
            s = next_s

            if args['visualize']:
                env.render()
            steps += 1

        print('Test reward:', ep_reward)
