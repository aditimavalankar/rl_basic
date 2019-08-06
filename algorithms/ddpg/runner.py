from networks.network import ddpg_actor, ddpg_critic
from algorithms.ddpg.ddpg import train, test
from utils.helper import load_checkpoint
from torch.optim import Adam


def run(args):
    env = args['env']
    device = args['device']
    input_shape = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = ddpg_actor(input_shape, action_dim).to(device)
    critic = ddpg_critic(input_shape, 1, action_dim).to(device)
    # critic = mlp(input_shape, 1, activation='relu').to(device)
    target_actor = ddpg_actor(input_shape, action_dim).to(device)
    target_critic = ddpg_critic(input_shape, 1, action_dim).to(device)

    target_actor.set_weights_like(actor)
    target_critic.set_weights_like(critic)

    actor_optimizer = Adam(actor.parameters(), lr=args['actor_lr'])
    critic_optimizer = Adam(critic.parameters(), lr=args['critic_lr'],
                            weight_decay=0)

    total_steps = 0
    total_episodes = 0

    if args['ckpt_file']:
        params = {'params': ['total_steps',
                             'total_episodes',
                             'actor',
                             'target_actor',
                             'critic',
                             'target_critic',
                             'actor_optimizer',
                             'critic_optimizer',
                             'state_info',
                             'reward_info']}
        (total_steps,
         total_episodes,
         actor_weights,
         target_actor_weights,
         critic_weights,
         target_critic_weights,
         actor_optimizer_params,
         critic_optimizer_params,
         state_params,
         reward_params) = load_checkpoint(args['ckpt_file'], **params)

        actor.set_weights(actor_weights)
        target_actor.set_weights(target_actor_weights)
        critic.set_weights(critic_weights)
        target_critic.set_weights(target_critic_weights)
        actor_optimizer.load_state_dict(actor_optimizer_params)
        critic_optimizer.load_state_dict(critic_optimizer_params)
        args['state_info'] = state_params
        args['reward_info'] = reward_params

    args['total_steps'] = total_steps
    args['total_episodes'] = total_episodes

    if not args['test_only']:
        trained_vars = train(actor, target_actor, critic, target_critic,
                             actor_optimizer, critic_optimizer, args)
        actor = trained_vars['actor']

    test(actor, args)
