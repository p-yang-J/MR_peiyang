import numpy as np
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import BCQ
from sklearn.ensemble import RandomForestRegressor
import torch
from d3rlpy.preprocessing import MinMaxActionScaler
# Assuming that user_feedback is a 1D array
user_feedback = np.load('feedbacks.npy')
action_scaler = MinMaxActionScaler(minimum=np.array([-3000,-3000,-4000,-3000]),maximum=np.array([3000,3000,4000,3000]))

dataset = MDPDataset(
        observations=np.load('observations.npy'),
        actions=np.load('actions.npy'),
        rewards=np.load('rewards.npy'),
        terminals=np.load('terminals.npy'),
    )
a = np.load('observations.npy')
# Load a pretrained BCQ model
dqn2 = BCQ()
dqn2.build_with_dataset(dataset)
dqn2.load_model('bcq_711.pt')


# Get predicted actions
predicted_actions = dqn2.predict(a)
predicted_actions = torch.tensor(predicted_actions, dtype=torch.float32)
predicted_actions = action_scaler.reverse_transform(predicted_actions)
# Save predictions to a file for later analysis
np.save('predicted_actions.npy', predicted_actions)
predicted_a = np.load('predicted_actions.npy')
np.savetxt('array.txt',predicted_a)
print(predicted_a)
# x_batch = [100, 100, 100, 100, 10, 100]
# x_array = np.array(x_batch)
# x_array = x_array.reshape(1, -1)
# actions_new = dqn2.predict(x_array)[0]
# # 将其转换为 tensor
# actions_new_tensor = torch.tensor(actions_new, dtype=torch.float32)
# # 使用 `reverse_transform` 方法来恢复原始的动作值
# action_scaler = MinMaxActionScaler(minimum=[-3000, -3000, -4000], maximum=[3000, 3000, 4000])
# actions_original = action_scaler.reverse_transform(actions_new_tensor)
# # 如果需要，你可以将结果再转换回 numpy 数组
# actions_original = actions_original.numpy()
# scale_addx = actions_original[0]
# scale_addy = actions_original[1]
# scale_addz = actions_original[2]
# print(scale_addz,scale_addy,scale_addx)

