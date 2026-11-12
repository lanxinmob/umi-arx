> 基于 [real-stanford/umi-arx 的 uva 分支](https://github.com/real-stanford/umi-arx/tree/uva)，增加 ARX X5 图像预处理、绝对关节动作执行、轨迹衔接及人工/策略切换。
>
> `absolute_joint` checkpoint 使用配套 SDK 的 `communication/zmq_joint_server.py`，动作是六关节弧度与夹爪米制宽度；笛卡尔策略使用对应末端控制服务。执行入口为 `scripts/eval_arx5.py`，参数见 `--help`。MCP 扩展尚有待解决的实机问题。

---

Please follow [This Notion Site](https://yihuai.notion.site/UMI-ARX-Cup-Task-Deployment-Tutorial-112e4c9a5c4980f1ac20fe4da261d9a1?pvs=4) for detailed deployment instructions.
