import torch
import torch.optim as optim

import globals as g
from replay_memory import ReplayMemory, Transition


def optimize(network: torch.nn.Module, memory: ReplayMemory):
    if len(memory) < g.BATCH_SIZE:
        return

    # Sample a batch of transitions
    transitions = memory.sample(g.BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=g.DEVICE,
        dtype=torch.bool,
    )

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = network(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next state
    next_state_values = torch.zeros(g.BATCH_SIZE, device=g.DEVICE)

    with torch.no_grad():
        next_state_values[non_final_mask] = network(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * g.GAMMA) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Finally, optimize
    optimizer = optim.RMSprop(network.parameters(), lr=g.LR)
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(network.parameters(), 100)
    optimizer.step()
