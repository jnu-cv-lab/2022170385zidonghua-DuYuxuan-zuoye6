计算机视觉 Lab 07: 局部特征检测与匹配 (ORB & SIFT)

这是计算机视觉课程第七次实验的代码和结果记录。本次实验主要对比了 ORB 与 SIFT 两种特征提取算法，并结合 RANSAC 实现了目标物体在复杂场景下的透视定位。

目录结构

为了避免代码和生成的图片混淆，本次实验按任务进行了分文件夹管理：

task1/：ORB 关键点与描述子提取。
task2/：基于 BFMatcher 的 ORB 初始特征匹配（Hamming距离 + crossCheck）。
task3/：使用 RANSAC 算法估算单应性矩阵（Homography）并剔除错误连线（外点）。
task4/：目标定位。将模板图的角点投影到场景图中，并绘制了绿色追踪框。
task5/：参数对比实验脚本。测试了 `nfeatures` 分别为 500、1000、2000 时对匹配数量和内点比例的影响。
task_bonus/：选做任务（SIFT 特征匹配）。使用 L2范数 + KNN + Lowe's Ratio Test 筛选匹配点，并绘制了红色的定位框。
根目录存放了原始的测试用例 `box.png` 和 `box_in_scene.png`。

运行环境
操作系统: Windows Subsystem for Linux (WSL) - Ubuntu
语言: Python
核心依赖库: `opencv-python`, `numpy`

快速运行
每个任务文件夹下都包含独立的 Python 脚本，直接进入对应目录运行即可。
例如，运行任务四的目标定位：
bash
cd task4
python task4_localization.py
其他详细信息请见上传的实验报告，里面有详细的说明。
