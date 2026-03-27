Github仓库：https://github.com/siyaoZHAO/MySoccerAgent

## 实验启动脚本与结果保存位置说明

本项目包含了基线（Baseline）以及多智能体（Multi-Agent）管线的不同实验执行脚本。所有脚本都已打包好运行环境与目录跳转，并可通过 `tmux` 维持后台运行。

### 1. 多智能体平台 (Multi-Agent Platform) 实验
这部分是调用 `platform_full_version.py`，使用完整的工具链进行多智能体推理与解答。

*   **Test 验证集运行脚本**: 
    *   **脚本路径**: `bash_platform_test.sh`
    *   **启动方式**: `bash bash_platform_test.sh`
    *   **结果保存位置**: 
        *   JSON 推理结果: 项目根目录下的 `metadata_ma_test_YYYYMMDD_HHMMSS.json`
        *   运行终端日志: `/log/metadata_ma_test_YYYYMMDD_HHMMSS.log`

*   **Challenge 挑战集运行脚本**:
    *   **脚本路径**: `bash_platform_challenge.sh`
    *   **启动方式**: `bash bash_platform_challenge.sh`
    *   **结果保存位置**: 
        *   JSON 推理结果: 项目根目录下的 `metadata_ma_challenge_YYYYMMDD_HHMMSS.json`
        *   运行终端日志: `/log/metadata_ma_challenge_YYYYMMDD_HHMMSS.log`

*   **计算 Accuracy 脚本**:
    *   **脚本路径**: `bash_acc.sh`
    *   **作用**: 当 test 的 JSON 推理完成后，用于计算准确率，需进入脚本修改对应的 `--output_file` 路径为您想要计算的 JSON 结果文件。

### 2. 基线 (Baseline) 实验
这部分存放于 `baseline/` 目录下，用于与多智能体系统做对比。

*   **Test 验证集基线运行脚本**:
    *   **脚本路径**: `baseline/exam_test.sh` 或 `baseline/exam_test_acc.sh`

*   **Challenge 挑战集基线运行脚本**:
    *   **脚本路径**: `baseline/exam_challenge.sh`

*   **基线结果保存位置**:
    *   均保存在 `baseline/` 文件夹内部，命名格式类似于 `metadata_test_YYYYMMDD_HHMMSS.json` 或 `metadata_YYYYMMDD_HHMMSS.json`。


