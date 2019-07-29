# 联合姿态先验的人体精确解析双分支网络模型
主要基于detectron框架，具体安装详见https://github.com/facebookresearch/Detectron
主要改进：
1、引入pose-attention模块，提高人体四肢部件的分割精度
2、使用检测分枝提高小目标部件的分割精度

注意：
使用openpose body25提取人体姿态信息

# maskrcnn_body25
