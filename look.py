#
# import numpy as np
# import os
#
# # 原npy文件路径
# npy_path = "/home/zxb/GuidedContrast/exp/s3dis/semantic_semi/semseg_run1_5_semi_s3dis/result/iter2000/val/semantic/Area_3_conferenceRoom_19.npy"
# # 输出txt路径（与npy同目录，仅扩展名不同）
# txt_path = os.path.splitext(npy_path)[0] + ".txt"
#
# # 读取npy文件
# data = np.load(npy_path)
#
# # 保存为txt文件
# np.savetxt(txt_path, data, fmt="%.8f" if np.issubdtype(data.dtype, np.floating) else "%d")
#
# print(f"已将 {npy_path} 转换为 {txt_path}")
#
#
# import torch
# import numpy as np
# import os
#
# pth_path = "/home/zxb/GuidedContrast/dataset/s3dis/s3dis/Area_3_conferenceRoom_19.pth"
# # 输出txt路径（同名，仅扩展名不同）
# txt_path = os.path.splitext(pth_path)[0] + ".txt"
#
# # 读取pth文件（通常为tuple或list，可能包含多个数组）
# data = torch.load(pth_path)
#
# # 检查内容类型
# if isinstance(data, (tuple, list)):
#     # 将每个数组分别存为txt，多数组时加后缀
#     for idx, arr in enumerate(data):
#         arr = arr.cpu().numpy() if hasattr(arr, "cpu") else np.array(arr)
#         # 如果只有一个数组则直接用主txt名，否则加编号
#         part_txt_path = txt_path if len(data) == 1 else txt_path.replace(".txt", f"_{idx+1}.txt")
#         np.savetxt(part_txt_path, arr, fmt="%.8f" if np.issubdtype(arr.dtype, np.floating) else "%d")
#         print(f"已将 {pth_path} 的第{idx+1}部分转换为 {part_txt_path}")
# else:
#     arr = data.cpu().numpy() if hasattr(data, "cpu") else np.array(data)
#     np.savetxt(txt_path, arr, fmt="%.8f" if np.issubdtype(arr.dtype, np.floating) else "%d")
#     print(f"已将 {pth_path} 转换为 {txt_path}")

# import numpy as np
#
# # 文件路径
# semantic_txt = "/home/zxb/GuidedContrast/exp/s3dis/semantic_semi/semseg_run1_5_semi_s3dis/result/iter2000/val/semantic/Area_3_conferenceRoom_19.txt"
# data_txt = "/home/zxb/GuidedContrast/dataset/s3dis/s3dis/Area_3_conferenceRoom_19_1.txt"
# out_txt = "/home/zxb/GuidedContrast/area3_19_2000.txt"
#
# # 加载数据
# semantic_col = np.loadtxt(semantic_txt)
# data = np.loadtxt(data_txt)
#
# # 保证semantic_col是一维
# if semantic_col.ndim > 1:
#     semantic_col = semantic_col[:, 0]
#
# # 检查行数一致
# if data.shape[0] != semantic_col.shape[0]:
#     raise ValueError(f"行数不一致: {data.shape[0]} vs {semantic_col.shape[0]}")
#
# # 追加新列
# result = np.column_stack([data, semantic_col])
#
# # 保存
# np.savetxt(out_txt, result, fmt="%.8f" if np.issubdtype(result.dtype, np.floating) else "%d")
#
# print(f"已生成 {out_txt}，新文件共有 {result.shape[1]} 列。")

import re
import os


def find_max_miou(log_file_path):
    max_miou = 0.0
    max_line_number = 0
    max_iteration = 0
    found_evaluation = False
    pattern = r'iter: (\d+)/\d+,.*?miou: (\d+\.\d+)'
    line_number = 0
    results = []  # 存储所有符合条件的记录

    try:
        if not os.path.exists(log_file_path):
            raise FileNotFoundError(f"文件未找到: {log_file_path}")

        with open(log_file_path, 'r') as file:
            for line in file:
                line_number += 1

                # 检测评估开始标记
                if "---------------------" in line:
                    found_evaluation = True
                    continue

                # 如果是评估标记后的第一行
                if found_evaluation:
                    found_evaluation = False  # 重置标记
                    match = re.search(pattern, line)

                    if match:
                        iter_num = int(match.group(1))
                        miou_val = float(match.group(2))
                        results.append((line_number, iter_num, miou_val))  # 添加到结果列表
                        # # 检查迭代数是否能被250整除
                        # if iter_num % 250 == 0:
                        #     results.append((line_number, iter_num, miou_val))
                        # 更新最大值
                        if miou_val > max_miou:
                            max_miou = miou_val
                            max_line_number = line_number
                            max_iteration = iter_num

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return None, None, None, []

    return max_miou, max_line_number, max_iteration, results


if __name__ == "__main__":
    log_file = "/home/zxb/GuidedContrastrandnew/exp/s3dis/semantic_semi/semseg_run1_10_baseline_s3dis/train-20250803_122249.log"
    max_miou, max_line, max_iter, all_results = find_max_miou(log_file)

    if max_miou > 0:
        print(f"最大miou值: {max_miou:.4f}")
        print(f"所在行号: {max_line}")
        print(f"对应迭代: {max_iter}")
        print("\n所有符合条件的记录:")
        print("{:<10} {:<10} {:<10}".format("行号", "迭代", "miou"))
        print("-" * 30)
        for line_num, iter_num, miou_val in all_results:
            print(f"{line_num:<10} {iter_num:<10} {miou_val:.4f}")
    else:
        print("未找到符合条件的记录")


import re
import os

# def find_max_miou(log_file_path):
#     max_miou = 0.0
#     max_line_number = 0
#     max_iteration = 0
#     results = []  # 存储所有符合条件的记录
#     pattern = r'iter: (\d+)/\d+.*?val_mIoU \(avg over 5 samples\): (\d+\.\d+)'
#
#     try:
#         if not os.path.exists(log_file_path):
#             raise FileNotFoundError(f"文件未找到: {log_file_path}")
#
#         with open(log_file_path, 'r') as file:
#             for line_number, line in enumerate(file, 1):
#                 match = re.search(pattern, line)
#                 if match:
#                     iter_num = int(match.group(1))
#                     miou_val = float(match.group(2))
#                     results.append((line_number, iter_num, miou_val))
#                     # 更新最大值
#                     if miou_val > max_miou:
#                         max_miou = miou_val
#                         max_line_number = line_number
#                         max_iteration = iter_num
#
#     except Exception as e:
#         print(f"处理文件时出错: {str(e)}")
#         return None, None, None, []
#
#     return max_miou, max_line_number, max_iteration, results
#
# if __name__ == "__main__":
#     log_file = "/home/zxb/GuidedContrastrandnew/exp/s3dis/semantic_semi/semseg_run1_100_baseline_s3dis/train-20250725_023717.log"
#     max_miou, max_line, max_iter, all_results = find_max_miou(log_file)
#
#     if max_miou > 0:
#         print(f"最大miou值: {max_miou:.4f}")
#         print(f"所在行号: {max_line}")
#         print(f"对应迭代: {max_iter}")
#         print("\n所有符合条件的记录:")
#         print("{:<10} {:<10} {:<10}".format("行号", "迭代", "miou"))
#         print("-" * 30)
#         for line_num, iter_num, miou_val in all_results:
#             print(f"{line_num:<10} {iter_num:<10} {miou_val:.4f}")
#     else:
#         print("未找到符合条件的记录")