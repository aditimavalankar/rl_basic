import numpy as np
from algorithms.ddpg.noise import OrnsteinUhlenbeckActionNoise
from algorithms.ddpg.replay_buffer import ReplayBuffer
from tensorboardX import SummaryWriter
import os
from utils.helper import save_checkpoint, np2tensor
from utils.runningmeanstd import RunningMeanStd
import torch.nn as nn
# from torchviz import make_dot


def train(actor, target_actor, critic, target_critic,
          actor_optimizer, critic_optimizer, args):
    """Train the DDPG learner."""

    env = args['env']
    device = args['device']
    total_steps = args['total_steps']
    total_episodes = args['total_episodes']
    action_dim = env.action_space.shape[0]
    replay = ReplayBuffer(max_size=1000000)
    reward_buffer = []
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
    s = state_rms.normalize(s)
    done = False
    ep_reward = 0

    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim),
                                                sigma=0.2 *
                                                np.ones(action_dim))

    while total_steps < args['n_training_steps']:

        # while True:
        actor.eval()
        a = actor(np2tensor(s, device)) + np2tensor(action_noise(), device)
        a = a.clamp(-1, 1)
        actor.train()
        # a = actor(np2tensor(s, device))
        a = a.data.to('cpu').numpy()
        next_s, r, done, _ = env.step(a)
        ep_reward += r
        next_s = state_rms.normalize(next_s)
        # r = return_rms.normalize(r)
        experience = (s, a, r, next_s, int(done))
        replay.update(experience)
        s = next_s
        total_steps += 1

        if done:
            reward_buffer.append(ep_reward)
            if len(reward_buffer) > 100:
                reward_buffer = reward_buffer[-100:]
            writer['iter'] = total_steps
            writer['writer'].add_scalar('data/ep_reward',
                                        ep_reward,
                                        total_steps)
            writer['writer'].add_scalar('data/last_100_ret',
                                        np.array(reward_buffer).mean(),
                                        total_steps)
            # writer['writer'].add_scalar('data/ep_len',
            #                             ep_len,
            #                             total_steps)
            s = env.reset()
            ep_reward = 0
            total_episodes += 1
            action_noise.reset()

        if len(replay) < args['batch_size']:
            continue

        for i in range(args['updates_per_step']):
            actor_loss = 0
            critic_loss = 0
            minibatch = replay.sample(size=args['batch_size'])

            exp_s = np2tensor(minibatch['state'], device)
            exp_next_s = np2tensor(minibatch['next_state'], device)
            exp_a = np2tensor(minibatch['action'], device)
            exp_r = np2tensor(minibatch['reward'], device)
            exp_done = np2tensor(minibatch['done'], device)

            target_next_a = target_actor(exp_next_s)
            # target_next_q = return_rms.denormalize(target_critic(exp_next_s,
            #                                                      target_next_a)
            #                                        ).detach()
            target_next_q = target_critic(exp_next_s, target_next_a)
            target_q = exp_r + (1 - exp_done) * args['gamma'] * target_next_q

            pred_q = critic(exp_s, exp_a)

            critic_optimizer.zero_grad()
            # critic_loss = ((pred_q - target_q) ** 2) / args['batch_size']
            critic_loss = loss_fn(pred_q, target_q)
            critic_loss.backward()
            critic_optimizer.step()

            # target_q = return_rms.normalize(target_q)
            # actor_loss -= return_rms.denormalize(critic(exp_s, actor(exp_s)))
            actor_optimizer.zero_grad()
            actor_loss = -critic(exp_s, actor(exp_s)).mean()
            actor_loss.backward()
            actor_optimizer.step()

            target_actor.set_weighted_weights(actor, args['tau'])
            target_critic.set_weighted_weights(critic, args['tau'])

            writer['iter'] = total_steps * args['updates_per_step'] + i + 1
            writer['writer'].add_scalar('data/actor_loss',
                                        actor_loss,
                                        total_steps * args['updates_per_step'] + i + 1)
            writer['writer'].add_scalar('data/critic_loss',
                                        critic_loss,
                                        total_steps * args['updates_per_step'] + i + 1)

        # critic_loss /= args['batch_size']
        # actor_loss /= args['batch_size']

        # for (n, p) in actor.named_parameters():
        #     if 'final' not in n:
        #         actor_loss += args['l2_reg'] * torch.norm(p.flatten(), p=2)
        #
        # for (n, p) in critic.named_parameters():
        #     if 'final' not in n:
        #         critic_loss += args['l2_reg'] * torch.norm(p.flatten(), p=2)

        print('Steps: %d, Avg. reward: %f' % (total_steps,
                                              np.array(reward_buffer).mean()))

        # actor_optimizer.zero_grad()
        # critic_optimizer.zero_grad()
        #
        # actor_loss.backward()
        # critic_loss.backward()
        #
        # actor_optimizer.step()
        # critic_optimizer.step()

        # target_actor.set_weighted_weights(actor, args['tau'])
        # target_critic.set_weighted_weights(critic, args['tau'])

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


def test(actor, args):
    env = args['env']

    for ep in range(args['n_test_episodes']):
        s = env.reset()
        ep_reward = 0
        done = False

        while not done:
            s = np2tensor(s)
            a = actor(s).to('cpu').numpy()
            next_s, r, done, _ = env.step(a)
            ep_reward += r
            if args['visualize']:
                env.render()

        print('Episode %d reward: %f' % (ep + 1, ep_reward))
