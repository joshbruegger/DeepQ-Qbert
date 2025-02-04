import gc

import torch
import torch.optim as optim

import globals as g
from replay_memory import ReplayMemory, Transition


def optimize(network: torch.nn.Module, memory: ReplayMemory):
    if len(memory) < g.BATCH_SIZE:
        return None, None  # Return None if batch size is insufficient

    global optimizer
    if "optimizer" not in globals():
        optimizer = optim.RMSprop(network.parameters(), lr=g.LR)

    # Clear memory before starting
    gc.collect()
    torch.cuda.empty_cache()

    transitions = memory.sample(g.BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Combine tensor operations to reduce overhead
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=g.DEVICE,
        dtype=torch.bool,
    )

    # Pre-allocate next_state_values and fill only non-final states
    next_state_values = torch.zeros(g.BATCH_SIZE, device=g.DEVICE)
    if non_final_mask.any():  # Only process if there are non-final states
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        with torch.no_grad():
            next_values = network(non_final_next_states).max(1).values
            next_state_values[non_final_mask] = next_values
            del next_values  # Explicitly delete to free memory

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = network(state_batch)
    q_values = state_action_values.gather(1, action_batch)
    max_q = state_action_values.max().item()  # Get the maximum Q-value

    # Clean up GPU memory
    del state_action_values
    torch.cuda.empty_cache()

    expected_state_action_values = (next_state_values * g.GAMMA) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(q_values, expected_state_action_values.unsqueeze(1))

    # Clean up intermediate tensors
    del next_state_values, expected_state_action_values
    torch.cuda.empty_cache()

    # Finally, optimize
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(network.parameters(), 100)
    optimizer.step()

    # Get values before cleanup
    loss_item = loss.item()

    # Final cleanup
    del loss, q_values, state_batch, action_batch, reward_batch
    torch.cuda.empty_cache()
    gc.collect()

    return loss_item, max_q  # Return both loss and max Q-value
