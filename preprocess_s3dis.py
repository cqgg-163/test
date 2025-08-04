# # filename: preprocess_s3dis.py
# import os
# import glob
# import numpy as np
# import torch
# from tqdm import tqdm
# import warnings
#
# # --- 参数配置 ---
# # 请根据您的实际路径进行修改
# DATA_ROOT = 'dataset'  # 包含 s3dis 文件夹的根目录
# SAVE_PATH = os.path.join(DATA_ROOT, 's3dis', 'processed_blocks')  # 保存预处理后数据块的路径
# BLOCK_SIZE = 1.0  # 切块大小 (米)
# BLOCK_STRIDE = 0.5  # 切块步长 (米)
# NUM_POINTS = 40960  # 每个块采样到的目标点数
#
# # 忽略numpy中的未来警告
# warnings.filterwarnings("ignore", category=FutureWarning, module='numpy')
#
#
# def room_to_blocks(points, labels, num_points_per_block, block_size=1.0, stride=1.0, random_sample=False):
#     """
#     将一个房间的点云切割成多个数据块
#     """
#     # 确保点云是 numpy array
#     points = np.array(points)
#     labels = np.array(labels)
#
#     limit = np.amax(points, 0)[0:3]
#
#     # 获取XY平面的切块数量
#     width = int(np.ceil((limit[0] - block_size) / stride)) + 1
#     depth = int(np.ceil((limit[1] - block_size) / stride)) + 1
#
#     blocks = []
#     block_labels = []
#
#     for index_y in range(depth):
#         for index_x in range(width):
#             # 计算当前块的边界
#             x_min = stride * index_x
#             x_max = x_min + block_size
#             y_min = stride * index_y
#             y_max = y_min + block_size
#
#             # 筛选出在当前块内的点
#             point_indices = np.where(
#                 (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
#                 (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
#             )[0]
#
#             # 如果块内点太少，则跳过
#             if point_indices.shape[0] < 100:
#                 continue
#
#             num_points_in_block = point_indices.shape[0]
#
#             # --- 采样点 ---
#             if random_sample:
#                 # 随机重复采样或下采样到固定点数
#                 choice = np.random.choice(num_points_in_block, num_points_per_block, replace=True)
#             else:
#                 # 如果点数不够，则重复采样；如果点数够，则随机下采样
#                 choice = np.random.choice(num_points_in_block, num_points_per_block,
#                                           replace=(num_points_in_block < num_points_per_block))
#
#             selected_points = points[point_indices[choice], :]
#             selected_labels = labels[point_indices[choice]]
#
#             # --- 关键步骤：逐块局部归一化 ---
#             # 归一化后的点：(normalized_xyz, original_rgb, original_xyz)
#             # 这种保存原始坐标的方式，在某些高级任务中有用，这里我们简化
#             block_points_normalized = np.zeros_like(selected_points)
#
#             # 计算块的中心
#             center = np.mean(selected_points[:, :3], axis=0)
#
#             # 坐标归一化
#             block_points_normalized[:, :3] = selected_points[:, :3] - center
#             # 颜色归一化 (可选，但推荐)
#             block_points_normalized[:, 3:6] = selected_points[:, 3:6] / 255.0
#
#             blocks.append(block_points_normalized)
#             block_labels.append(selected_labels)
#
#     return np.array(blocks), np.array(block_labels)
#
#
# def main():
#     if not os.path.exists(SAVE_PATH):
#         os.makedirs(SAVE_PATH)
#
#     # 获取所有原始数据文件路径
#     room_file_paths = sorted(glob.glob(os.path.join(DATA_ROOT, 's3dis', 's3dis', '*.pth')))
#
#     print(f"找到 {len(room_file_paths)} 个原始房间文件。")
#     print(f"预处理后的数据块将保存到: {SAVE_PATH}")
#
#     for room_path in tqdm(room_file_paths, desc="Processing Rooms"):
#         try:
#             # 加载原始数据
#             data = torch.load(room_path)
#             # data[0] 是 xyz+rgb, data[1] 是 label
#             points, labels = data[0], data[1]
#
#             # 将房间切割成块并进行局部归一化
#             processed_blocks, processed_labels = room_to_blocks(
#                 points, labels,
#                 num_points_per_block=NUM_POINTS,
#                 block_size=BLOCK_SIZE,
#                 stride=BLOCK_STRIDE,
#                 random_sample=False
#             )
#
#             # 将每个块保存为独立的 .npy 文件
#             room_name = os.path.basename(room_path).replace('.pth', '')
#             for i in range(processed_blocks.shape[0]):
#                 save_filename = f"{room_name}_block_{i:04d}.npy"
#                 # 将点云数据和标签保存在同一个文件中
#                 np.save(os.path.join(SAVE_PATH, save_filename), {
#                     'points': processed_blocks[i],
#                     'labels': processed_labels[i]
#                 })
#
#         except Exception as e:
#             print(f"\n处理文件 {room_path} 时出错: {e}")
#             continue
#
#     print("\n所有房间预处理完成！")
#
#
# if __name__ == '__main__':
#     main()

# filename: preprocess_s3dis.py
# filename: preprocess_s3dis.py (最终修正版)
# filename: preprocess_s3dis.py (最终正确版)
import os
import glob
import numpy as np
import torch
from tqdm import tqdm
import warnings

# --- 参数配置 (根据您代码中的信息设定) ---
DATA_ROOT = 'dataset'
# 输入：您第一步处理后生成的、包含整个房间的文件目录
# 注意：根据您最初的代码，它是在 'dataset/s3dis/s3dis' 目录下寻找文件
INPUT_DATA_PATH = os.path.join(DATA_ROOT, 's3dis', 's3dis1')
# 输出：切块后，用于最终训练的数据保存目录
SAVE_PATH = os.path.join(DATA_ROOT, 's3dis', 's3dis')
BLOCK_SIZE = 1.0
BLOCK_STRIDE = 0.5
NUM_POINTS = 40960

warnings.filterwarnings("ignore", category=FutureWarning, module='numpy')


def room_to_blocks(points, labels, num_points_per_block, block_size, stride):
    """
    只进行切块和采样，不执行任何归一化操作。
    """
    # 假设输入的 points 是 (N, 6) 的 xyzrgb 数据
    coords = points[:, :3]
    colors = points[:, 3:6]

    limit = np.amax(coords, axis=0)
    width = int(np.ceil((limit[0] - block_size) / stride)) + 1
    depth = int(np.ceil((limit[1] - block_size) / stride)) + 1

    output_blocks = []

    for y in range(depth):
        for x in range(width):
            x_min, x_max = stride * x, stride * x + block_size
            y_min, y_max = stride * y, stride * y + block_size

            mask = (coords[:, 0] >= x_min) & (coords[:, 0] < x_max) & \
                   (coords[:, 1] >= y_min) & (coords[:, 1] < y_max)

            if np.sum(mask) < 100:
                continue

            block_coords = coords[mask, :]
            block_colors = colors[mask, :]
            block_labels = labels[mask]

            n_points = len(block_coords)
            if n_points == 0: continue

            # 按需采样到固定点数
            if n_points > num_points_per_block:
                choice = np.random.choice(n_points, num_points_per_block, replace=False)
            else:
                choice = np.random.choice(n_points, num_points_per_block, replace=True)

            # 将采样后的数据块作为一个三元组添加到列表中
            output_blocks.append((block_coords[choice], block_colors[choice], block_labels[choice]))

    return output_blocks


def main():
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    room_file_paths = sorted(glob.glob(os.path.join(INPUT_DATA_PATH, '*.pth')))
    if not room_file_paths:
        print(f"错误: 在输入目录 '{INPUT_DATA_PATH}' 中没有找到任何 .pth 文件。请确认您的原始房间数据文件位置正确。")
        return

    print(f"找到 {len(room_file_paths)} 个原始房间文件，将进行切块...")

    for room_path in tqdm(room_file_paths, desc="Blocking Rooms"):
        try:
            # 加载您的“整间房”数据
            # 假设其格式为 (xyzrgb_numpy_array, labels_numpy_array)
            data = torch.load(room_path)
            points_data, labels_data = data[0], data[1]

            # 执行切块和采样
            processed_blocks = room_to_blocks(points_data, labels_data, NUM_POINTS, BLOCK_SIZE, BLOCK_STRIDE)

            room_name = os.path.basename(room_path).replace('.pth', '')
            for i, (block_coords, block_colors, block_labels) in enumerate(processed_blocks):
                save_filename = f"{room_name}_block_{i:04d}.pth"

                # ================= 关键修正 =================
                # 保存为与您原始代码完全兼容的 (坐标, 颜色, 标签) 三元组Numpy数据
                # ============================================
                torch.save((block_coords, block_colors, block_labels), os.path.join(SAVE_PATH, save_filename))

        except Exception as e:
            print(f"\n处理文件 {room_path} 时出错: {e}")
            continue

    print(f"\n所有房间切块完成！数据块保存在: {SAVE_PATH}")


if __name__ == '__main__':
    main()