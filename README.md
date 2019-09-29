```
util
.
|-- backup_and_old_version: 备份与旧版本脚本目录
|-- check_cleaned_data.py: 检查清洗的数据是否有问题
|-- clean_raw_data.py: 清洗数据（北京）
|-- clean_raw_data_TJ.py: 清洗数据（天津）
|-- clean_raw_data_TS.py: 清洗数据（唐山）
|-- compute_magic_tri_ratio.py：计算三角比
|-- compute_proposition_accurate_rate.py：验证在给定time interval和space interval的情况下，得到的stay segments满足在一个地方逗留的假设
|-- compute_stats_for_all_parts.py: label完数据后每个part都有对应的统计文件，合并这些统计文件来统计一些相关值，如 global/local sparsity, travel label ratio 等
|-- compute_stats_for_all_parts_uniform.PY:同上，只是BIN不一样
|-- compute_stats_for_all_parts_uniform_50000.PY:同上，只是BIN不一样
|-- compute_stats_simulation.py：对simulation的数据进行统计
|-- get_large_travel_sample.py：获取travel占比较大的轨迹数据（前5000part）
|-- get_large_travel_sample2.py：获取travel占比较大的轨迹数据（后5000part）
|-- get_large_travel_sample_simulation.py：同上，获取simulation的数据
|-- get_large_travel_sample_TJ(TS).py：同上，获取天津（唐山）的数据
|-- get_random_sample_data.py：随机sample数据
|-- get_random_sample_data_simulation.py：同上，针对simulation数据
|-- get_random_sample_data_TJ.py：同上，针对天津数据
|-- get_random_sample_data_TS.py：同上，针对唐山数据
|-- get_resample_data.py：对label占比最大的sparsity数据进行resample
|-- get_resample_data_by_percent.py: 以get_resample_data.py生成的数据为输入，按10%-100% 10个比例共sample出十份数据
|-- get_resample_data.py: 获得 resample 实验数据，将label后的数据按local sparisty的值分成100份，取local sparsity最大的那个区间，从中sample两万人数据 (10k train + 10k test)，另外这些数据要去除没有label的记录。
|-- get_sample_data.py: 获得 sample 实验数据, 将label后的数据按global sparsity或local sparsity分成100份，从后往前遍历合成10份，保证每个区间宽度尽量接近且每份trajectory数量不小于2万 (10k train + 10k test)
|-- get_sample_data_simulator.py：同上，针对simulation数据
|-- get_unobserved_experiment_data.py: 获得 unobserved 实验数据，对于每条trajectory, encoder部分sample一部分数据, decoder sample剩下的不超过200个记录
|-- label_and_compute_stats.py: 对数据打label并输出对应统计文件，用法 python3 label_and_compute_stats.py [minute] [space] [if_print_original_format], 其中[if_print_orignal_format]选项为0或1, 为0时每条记录输出一行，为1时相同id的记录输出一行，记录间以'|'分割，默认值为1。
|-- label_and_compute_stats_single.py：同上，针对单一文件
|-- label_and_compute_stats_TJ.py：同上，针对天津数据
|-- label_and_compute_stats_TS.py：同上，针对唐山数据
|-- label_and_evaluate.py:对simulation数据进行label，并统计各项指标
|-- process_path_length_data.py: 统计处理 path_length 数据
|-- sds_algorithm.py：sds算法实现
|--summarize_proposition_accurate_rate.py：合并compute_proposition_accurate_rate.py的统计值，得到总的值。
|-- trajectory_generator.py：生成simulation数据
|-- feature_extraction.py: 特征提取，把原始记录转换为模型输入需要的格式
```

```
stats_ML_models
.
|-- model_feature.py: 窗口模型, 参数见 python3 model1.py -h
|-- model_voting.py: voting模型
|-- model_hmm.py: HMM模型
|-- *.sh: 跑模型的shell脚本
|-- process_data_for_model_feature.py: 从原始数据抽取features提供给model_feature, 参数见 python3 process_data_for_model_feature.py -h
```

```
model
.
|-- model_run.py: 运行模型
|-- TD_model.py: 模型定义文件（LSTM，ATT等）
|-- TD_reader.py: 输入数据处理文件
```

```
42:/datahouse/yurl/TalkingData/data
.
|-- BJ/TJ/TS_cleaned_data: 北京/天津/唐山清洗后原始数据
|-- default: （废弃）
|-- P3: 在清洗过的北京数据上跑了label后的数据
|-- P3-sample: 由 get_sample_data.py 生成的global sparsity与local sparsity的数据文件,格式为train/test-10000-{time}-{space}-{type}-{which part}, 其中type为1或2,分别代表global sparsity与local sparsity
|-- P3-resample: 废弃，不用看
|—- P4: 在清洗过的北京数据上跑了label, 但保留原来的数据(id,经纬度等)
|-- P4-resample: resample 实验
|-- P3-SS-BJ/TJ/TS: label后的数据
|-- P3-SS-BJ/TJ/TS-inputdata: sample得到的模型输入数据（训练集/测试集）
|-- P3-SS-BJ/TJ/TS-resample: resample实验数据
|-- P3-SS-BJ-dense: 密集数据集
```
