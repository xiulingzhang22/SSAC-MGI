from multiprocessing.spawn import import_main_path
import numpy as np
import torch
from torch.nn import functional as F

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common import logger as sblogger
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import explained_variance, update_learning_rate, polyak_update
from stable_baselines3.common.type_aliases import Schedule

from .buffers import RolloutBuffer

from torch.utils.tensorboard import SummaryWriter
import pdb


def clip_range_fn(_: float) -> float:
    return 0.2


ENT_COEF = 0.
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
N_EPOCHS = 10
TAU = 0.005


def ppo_train(
        policy: ActorCriticPolicy,
        rollout_buffer: RolloutBuffer,
        progress: float,
        lr_schedule_fn: Schedule,
        writer: SummaryWriter = None,
        episode: int = None,
        policy_name: str = None
    ) -> None:
    """
    This is copied from 'train' in PPO in stable baselines
    The major modification is use of 'interventions' to only learn from certain experiences.
    """
    # Update optimizer learning rate
    lr = lr_schedule_fn(progress)
    #TJ
    logger = sblogger.Logger('./', ['txt'])
    logger.record("train/learning_rate", lr)
    update_learning_rate(policy.optimizer, lr)

    # Compute current clip range
    clip_range = clip_range_fn(progress)

    entropy_losses, all_kl_divs = [], []
    pg_losses, value_losses = [], []
    clip_fractions = []

    # train for n_epochs epochs
    for _ in range(N_EPOCHS):
        approx_kl_divs = []
        # Do a complete pass on the rollout buffer
        for rollout_data in rollout_buffer.get(rollout_buffer.buffer_size):
            actions = rollout_data.actions
            if policy_name == 'intervenge':
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            values, log_prob, entropy = policy.evaluate_actions(
                rollout_data.observations, actions)
            values = values.flatten()
            # Normalize advantage
            advantages = rollout_data.advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            # ratio = torch.exp(log_prob - rollout_data.old_log_prob)
            ratio = torch.exp(rollout_data.interventions * (log_prob - rollout_data.old_log_prob))

            # clipped surrogate loss
            # policy_loss_1 = advantages * ratio
            policy_loss_1 = advantages * ratio * rollout_data.interventions
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            # Logging
            pg_losses.append(policy_loss.item())
            clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
            clip_fractions.append(clip_fraction)

            values_pred = values
            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses.append(value_loss.item())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            entropy_losses.append(entropy_loss.item())

            loss = policy_loss + ENT_COEF * entropy_loss + VF_COEF * value_loss


            # Optimization step
            policy.optimizer.zero_grad()
            loss.backward()
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            policy.optimizer.step()
            approx_kl_divs.append(torch.mean(
                rollout_data.old_log_prob - log_prob).detach().cpu().numpy())
            
        if writer != None and policy_name != None:
            writer.add_scalar(f'ActorLoss_{policy_name}/train', policy_loss, episode)
            writer.add_scalar(f'CriticLoss_{policy_name}/train', VF_COEF*value_loss, episode)

        all_kl_divs.append(np.mean(approx_kl_divs))

    explained_var = explained_variance(rollout_buffer.values.flatten(),
                                       rollout_buffer.returns.flatten())

    # Logs
    logger.record("train/entropy_loss", np.mean(entropy_losses))
    logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    logger.record("train/value_loss", np.mean(value_losses))
    logger.record("train/approx_kl", np.mean(approx_kl_divs))
    logger.record("train/clip_fraction", np.mean(clip_fractions))
    logger.record("train/loss", loss.item())
    logger.record("train/explained_variance", explained_var)
    if hasattr(policy, "log_std"):
        logger.record("train/std", torch.exp(policy.log_std).mean().item())

    logger.record("train/clip_range", clip_range)
    #logger.dump() 
    
def gae(rewards, values, dones, discount_rate, lam):
    masks = 1. - np.array(dones)
    next_value = values[-1]
    advantages = []
    advantage = 0
    for reward, value, mask in zip(reversed(rewards), reversed(values), reversed(masks)):
        td_error = reward + discount_rate * next_value * mask - value
        advantage = td_error + discount_rate * lam * advantage * mask
        advantages.insert(0, advantage.item())
        next_value = value
    return torch.tensor(advantages, dtype=torch.float)
#Mapping intervene_action range[-1,1] to [0,1]
def mapping_action(action, action_space_low, action_space_high):
    mapped_action = action_space_low + \
                    ((action_space_high - action_space_low) / 2) * (action + 1)
    return mapped_action
def sac_train(
    policy: SACPolicy,
    replay_buffer: ReplayBuffer, 
    log_ent_coef,
    ent_coef_optimizer,
    lr_schedule, 
    progress: float,
    gradient_steps: int,
    target_entropy,
    batch_size: int = 1000,
    gamma: int = 1.,
    tau: float = 0.005,
    writer: SummaryWriter = None,
    episode:int = None,
    policy_name:str = None,
    other_player_policy: SACPolicy = None,
    other_player_buffer: ReplayBuffer = None,
    # intervene_policy: SACPolicy = None,
    # intervene_buffer: ReplayBuffer = None
) -> None:
    """Copied from SAC in stable baselines"""
    # Update optimizers learning rate
    optimizers = [policy.actor.optimizer, policy.critic.optimizer, ent_coef_optimizer]

    for optimizer in optimizers:
        update_learning_rate(optimizer, lr_schedule(progress))

    ent_coef_losses, ent_coefs = [], []
    actor_losses, critic_losses = [], []

    for _ in range(gradient_steps):
        # Sample replay buffer
        # ! more data
        replay_data = replay_buffer.sample(batch_size)
        
        # if policy_name != 'intervene':
        #     other_replay_data = other_player_buffer.sample(batch_size)
  
        # Action by the current actor for the sampled state
        actions_pi, log_prob = policy.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        # Important: detach the variable from the graph
        # so we don't change it with other losses
        # see https://github.com/rail-berkeley/softlearning/issues/60
        ent_coef = torch.exp(log_ent_coef.detach())
        ent_coef_loss = -(log_ent_coef * (log_prob + target_entropy).detach()).mean()
        ent_coef_losses.append(ent_coef_loss.item())

        ent_coefs.append(ent_coef.item())

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None:
            ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            ent_coef_optimizer.step()

        with torch.no_grad():
            # Select action according to policy
            # next_actions, next_log_prob = policy.actor.action_log_prob(replay_data.next_observations)

            # if policy_name != 'intervene':
            #     next_intervene_actions = \
            #         intervene_policy.forward(replay_data.next_observations, deterministic=False)
            #     next_intervene = next_intervene_actions >= 0
            #     if policy_name == 'standard':
            #         standard_policy = policy
            #         safe_policy = other_player_policy
            #     else:
            #         standard_policy = other_player_policy
            #         safe_policy = policy
            #     standard_next_actions, standard_next_log_prob = standard_policy.actor.action_log_prob(replay_data.next_observations)
            #     safe_next_actions, safe_next_log_prob = safe_policy.actor.action_log_prob(replay_data.next_observations)
            #     # next_actions = torch.zeros_like(standard_next_actions).unsqueeze(1)
            #     # pdb.set_trace()
            #     # next_actions[:, 0] = standard_next_actions
            #     # next_actions[:, 1] = safe_next_actions
            #     # next_log_prob = safe_next_log_prob if policy_name == 'safe' else standard_next_log_prob
                
            #     for i in range(len(next_intervene)):
            #         # print('next_actions[i]:', next_actions[i])
            #         # print('safe_next_actions[i]:', safe_next_actions[i])
            #         next_actions[i] = safe_next_actions[i] if next_intervene[i, 0] else standard_next_actions[i]
            #         next_log_prob[i] = safe_next_log_prob[i] if next_intervene[i, 0] else standard_next_log_prob[i]
            #     print('next_actions:', next_actions)
            #     # raise Exception()
            # else:
            #     next_actions, next_log_prob = policy.actor.action_log_prob(replay_data.next_observations)

            next_actions, next_log_prob = policy.actor.action_log_prob(replay_data.next_observations)
            # intervene_action=[-1,1]-->[0,1]
            if policy_name == 'intervene':
                next_actions = mapping_action(next_actions, 0, 1)
            # Compute the next Q values: min over all critics targets
            next_q_values = torch.cat(policy.critic_target(replay_data.next_observations, next_actions), dim=1)
            # pdb.set_trace()
            # if policy_name == 'intervene':
            #     next_q_values = torch.cat(policy.critic_target(replay_data.next_observations, next_actions), dim=1)
            # else:
            #     next_other_actions = other_player_policy.forward(replay_data.next_observations, deterministic=True)
            #     pdb.set_trace()
            #     # ! 3 inputs？
            #     next_q_values = torch.cat(policy.critic_target(replay_data.next_observations, next_actions, next_other_actions), dim=1)
            #     pdb.set_trace()
                    
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        # ! use average  
        current_q_values = policy.critic(replay_data.observations, replay_data.actions.float())
        # if policy_name == 'intervene':
        #     current_q_values = policy.critic(replay_data.observations, replay_data.actions.float())
        # else:
        #     current_q_values = policy.critic(replay_data.observations, replay_data.actions.float(), other_replay_data.actions.float())

        # Compute critic loss
        critic_loss = 0.5 * sum([
            F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
        critic_losses.append(critic_loss.item())

        # Optimize the critic
        policy.critic.optimizer.zero_grad()
        critic_loss.backward()
        policy.critic.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Mean over all critic networks
        
        q_values_pi = torch.cat(policy.critic.forward(replay_data.observations, actions_pi), dim=1)
        # if policy_name == 'intervene':
        #     q_values_pi = torch.cat(policy.critic.forward(replay_data.observations, actions_pi), dim=1)
        # else:
        #     other_actions = other_player_policy.actor.forward(replay_data.observations, deterministic=True)
        #     q_values_pi = torch.cat(policy.critic.forward(replay_data.observations, actions_pi, other_actions), dim=1)
        min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        actor_losses.append(actor_loss.item())

        # Optimize the actor
        policy.actor.optimizer.zero_grad()
        actor_loss.backward()
        policy.actor.optimizer.step()

        # Update target networks
        polyak_update(policy.critic.parameters(), policy.critic_target.parameters(), tau)

        if writer != None and policy_name != None:
            writer.add_scalar(f'ActorLoss_{policy_name}/train', actor_loss, episode)
            writer.add_scalar(f'CriticLoss_{policy_name}/train', critic_loss, episode)

    logger = sblogger.Logger('./', ['txt'])
    logger.record("train/ent_coef", np.mean(ent_coefs))
    logger.record("train/actor_loss", np.mean(actor_losses))
    logger.record("train/critic_loss", np.mean(critic_losses))
    if len(ent_coef_losses) > 0:
        logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        
    return ent_coef

def sac_train_joint_action(
    policy: SACPolicy,
    replay_buffer: ReplayBuffer, 
    log_ent_coef,
    ent_coef_optimizer,
    lr_schedule, 
    progress: float,
    gradient_steps: int,
    target_entropy,
    batch_size: int = 1000,
    gamma: int = 1.,
    tau: float = 0.005,
    writer: SummaryWriter = None,
    episode:int = None,
    policy_name:str = None,
    sample_num:float = 100,
    other_player_policy: SACPolicy = None,
    other_player_buffer: ReplayBuffer = None,
    intervene_policy: SACPolicy = None,
    intervene_buffer: ReplayBuffer = None,
) -> None:
    """Copied from SAC in stable baselines"""
    # Update optimizers learning rate
    optimizers = [policy.actor.optimizer, policy.critic.optimizer, ent_coef_optimizer]

    for optimizer in optimizers:
        update_learning_rate(optimizer, lr_schedule(progress))

    ent_coef_losses, ent_coefs = [], []
    actor_losses, critic_losses = [], []

    for _ in range(gradient_steps):
        # Sample replay buffer
        # ! more data
        replay_data = replay_buffer.sample(batch_size)
        
        if policy_name != 'intervene':
            other_replay_data = other_player_buffer.sample(batch_size)
        
        pdb.set_trace()        
  
        # Action by the current actor for the sampled state
        actions_pi, log_prob = policy.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)


        # Important: detach the variable from the graph
        # so we don't change it with other losses
        # see https://github.com/rail-berkeley/softlearning/issues/60
        ent_coef = torch.exp(log_ent_coef.detach())
        ent_coef_loss = -(log_ent_coef * (log_prob + target_entropy).detach()).mean()
        ent_coef_losses.append(ent_coef_loss.item())

        ent_coefs.append(ent_coef.item())
        

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None:
            ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            ent_coef_optimizer.step()

        with torch.no_grad():
            # Select action according to policy
            # next_actions, next_log_prob = policy.actor.action_log_prob(replay_data.next_observations)

            if policy_name != 'intervene':
                next_intervene_actions = \
                    intervene_policy.forward(replay_data.next_observations, deterministic=False)
                # True of False bool
                next_intervene = next_intervene_actions >= 0
                if policy_name == 'standard':
                    standard_policy = policy
                    safe_policy = other_player_policy
                else:
                    standard_policy = other_player_policy
                    safe_policy = policy
                # TODO sample multiple times to get the average Q
                standard_next_actions, standard_next_log_prob = standard_policy.actor.action_log_prob(replay_data.next_observations)
                safe_next_actions, safe_next_log_prob = safe_policy.actor.action_log_prob(replay_data.next_observations)
                # next_actions = torch.cat((standard_next_actions, safe_next_actions), dim=1)
                # next_actions = torch.zeros(standard_next_actions.shape[0], 2, standard_next_actions.shape[1])              
                # next_actions[:, 0] = standard_next_actions
                # next_actions[:, 1] = safe_next_actions
                next_actions = safe_next_actions if policy_name == 'safe' else standard_next_actions
                next_log_prob = safe_next_log_prob if policy_name == 'safe' else standard_next_log_prob
                
                # pdb.set_trace()
                # for i in range(len(next_intervene)):
                #     print(i)
                #     # print('next_actions[i]:', next_actions[i])
                #     # print('safe_next_actions[i]:', safe_next_actions[i])
                #     next_actions[i] = safe_next_actions[i] if next_intervene[i, 0] else standard_next_actions[i]
                #     next_log_prob[i] = safe_next_log_prob[i] if next_intervene[i, 0] else standard_next_log_prob[i]
                # print('next_actions:', next_actions)
                # raise Exception()
            else:
                next_actions, next_log_prob = policy.actor.action_log_prob(replay_data.next_observations)

            # next_actions, next_log_prob = policy.actor.action_log_prob(replay_data.next_observations)

            # Compute the next Q values: min over all critics targets
            # next_q_values = torch.cat(policy.critic_target(replay_data.next_observations, next_actions), dim=1)
            if policy_name == 'intervene':
                next_q_values = torch.cat(policy.critic_target(replay_data.next_observations, next_actions), dim=1)
            else:
                # sample multiple times
                next_q_values_list = []
                while sample_num > 0:
                    next_other_actions ,_ = other_player_policy.actor.action_log_prob(replay_data.next_observations)
                    # next_other_actions = other_player_policy.forward(replay_data.next_observations, deterministic=True)
                    # pdb.set_trace()
                    # ! 3 inputs？
                    next_q_values = torch.cat(policy.critic_target(replay_data.next_observations, next_actions, next_other_actions), dim=1)
                    next_q_values_list.append(next_q_values)
                    sample_num -= 1

                    
            # ! min前还是min后
            for i in range(1, len(next_q_values_list)):
                for j in range(next_q_values_list[0].shape[0]):
                    for k in range(next_q_values_list[0].shape[1]):
                        next_q_values_list[0][j][k] += next_q_values_list[i][j][k]
            # pdb.set_trace()
            next_q_values_averge = torch.divide(next_q_values_list[0], sample_num)
            
            next_q_values, _ = torch.min(next_q_values_averge, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * gamma * next_q_values

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        # ! use average  
        # current_q_values = policy.critic(replay_data.observations, replay_data.actions.float())
        if policy_name == 'intervene':
            current_q_values = policy.critic(replay_data.observations, replay_data.actions.float())
        else:
            current_q_values = policy.critic(replay_data.observations, replay_data.actions.float(), other_replay_data.actions.float())

        # Compute critic loss
        critic_loss = 0.5 * sum([
            F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
        critic_losses.append(critic_loss.item())

        # Optimize the critic
        policy.critic.optimizer.zero_grad()
        critic_loss.backward()
        policy.critic.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Mean over all critic networks
        
        # q_values_pi = torch.cat(policy.critic.forward(replay_data.observations, actions_pi), dim=1)
        if policy_name == 'intervene':
            q_values_pi = torch.cat(policy.critic.forward(replay_data.observations, actions_pi), dim=1)
        else:
            other_actions = other_player_policy.actor.forward(replay_data.observations, deterministic=True)
            q_values_pi = torch.cat(policy.critic.forward(replay_data.observations, actions_pi, other_actions), dim=1)
        min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        actor_losses.append(actor_loss.item())

        # Optimize the actor
        policy.actor.optimizer.zero_grad()
        actor_loss.backward()
        policy.actor.optimizer.step()

        # Update target networks
        polyak_update(policy.critic.parameters(), policy.critic_target.parameters(), tau)

        if writer != None and policy_name != None:
            writer.add_scalar(f'ActorLoss_{policy_name}/train', actor_loss, episode)
            writer.add_scalar(f'CriticLoss_{policy_name}/train', critic_loss, episode)


    logger = sblogger.Logger('./', ['txt'])
    logger.record("train/ent_coef", np.mean(ent_coefs))
    logger.record("train/actor_loss", np.mean(actor_losses))
    logger.record("train/critic_loss", np.mean(critic_losses))
    if len(ent_coef_losses) > 0:
        logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        
    return ent_coef