from agent.DQN.dqn_agent import DQNAgent
from agent.PPO.ppo_agent import PPOAgent


def get_training_agent(agent_name: str = 'PPO'):
    if agent_name == 'DQN':
        agent = DQNAgent(
            state_dim=1098,
            lr=1e-3,
            gamma=0.99,
            epsilon=0.8,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            memory_size=10000,
            batch_size=64,
            target_update=100,
            is_training=True
        )
    elif agent_name == 'PPO':
        agent = PPOAgent(
            obs_dim=1098,
            action_dim=2,
            max_step=20000,
            gamma=0.99,
            lamb=0.95,
            lr=1e-4,
            clip_val=0.2,
            max_grad_norm=0.5,
            ent_weight=0.01,
            sample_n_epoch=4,
            sample_mb_size=64,
            is_training=True
        )
    else:
        raise NotImplementedError

    return agent


def get_valid_agent(agent_name: str = 'PPO'):
    if agent_name == 'DQN':
        agent = DQNAgent(
            state_dim=1098,
            lr=1e-3,
            gamma=0.99,
            epsilon=0.8,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            memory_size=10000,
            batch_size=64,
            target_update=100,
            is_training=False
        )
    elif agent_name == 'PPO':
        agent = PPOAgent(
            obs_dim=1098,
            action_dim=2,
            max_step=20000,
            gamma=0.99,
            lamb=0.95,
            lr=1e-4,
            clip_val=0.2,
            max_grad_norm=0.5,
            ent_weight=0.01,
            sample_n_epoch=4,
            sample_mb_size=64,
            is_training=False
        )
    else:
        raise NotImplementedError

    return agent
