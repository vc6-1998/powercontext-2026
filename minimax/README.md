# MiniMax SDD 上下文模块

为已有任务清单项目修复“截止日期当天的任务被漏掉”的缺陷。工程规格、源码、测试存入
真实 seekdb，按任务查询生成上下文；MiniMax 提供实际代码修改；PowerContext 保存并复用
任务进展和测试结果。入口为命令行，不需要另行部署网页或完整 Agent 框架。

## 当前状态

当前交付为准备阶段。尚未配置 MiniMax 账号及 API Key，没有生成真实模型修改记录，
因此不能声称六项全部通过或可以正式提交。模型接入就绪后，执行 `--generate` 完成剩余流程。
`--prepare-only` 不会修改业务实现，也不会制造模型使用记录。

## 环境与安装

- Linux x86_64，Python 3.12；Windows 使用现有 Ubuntu WSL。
- PowerContext 使用未修改的官方 commit `9db1753434162c0aa7c9f1189a1722e617880a52`。
- `requirements.txt` 固定依赖，包括嵌入式 `pylibseekdb==1.3.0.post4`。
- seekdb 保存工程资料；PowerContext 通过其公共 Memory API 使用独立 SQLite 文件保存任务状态。
  SQLite 只作为 PowerContext 自己的后端，不替代工程资料的 seekdb 存储和检索。

首次安装需要访问 GitHub 和 Python 包索引。运行时不需要 Docker 或重新编译 seekdb。

```bash
cd minimax
bash bootstrap.sh
```

已有 uv 时也可用 `uv venv --python 3.12 .venv`，然后
`uv pip install --python .venv/bin/python -r requirements.txt`；本次已验证该安装方式。

如果 Python 3.12 不在 PATH，使用 `PYTHON=/绝对路径/python3.12 bash bootstrap.sh`。
初始化数据从交付样例生成，不依赖作者的数据库、全局项目索引或个人文件。
代码快照和证据位于输出目录；运行数据在 `/tmp/sdd-*` 独立短目录中，实际路径记录在报告。
这避免嵌入式 seekdb 的 Unix socket 超过 Linux 路径长度限制。数据保留供检查，后续可清理本轮专用目录。
嵌入式 seekdb 同一目录同时只供一个演示进程使用；不要并发运行相同状态目录。

## 凭据

按官方[领取指南](https://ask.oceanbase.com/t/topic/35653137)注册 MiniMax 开放平台并申请参赛权益。
填写开放平台用户 ID，不是队名。额度是否可用以账号实际状态为准。

将 `.env.example` 复制为 `.env.local` 后，在本地填写 `MINIMAX_API_KEY`。
`MINIMAX_MODEL` 和 `MINIMAX_BASE_URL` 应对应账号实际可用的模型及区域。
默认地址和模型来自 [MiniMax OpenAI 兼容接口文档](https://platform.minimaxi.com/docs/api-reference/text-openai-api)。
程序默认读取本目录 `.env.local`；也可以通过 `--env-file /绝对路径/.env.local` 使用单独配置文件。
只有 `--generate` 会调用模型；不自动重试或循环修复。
密钥文件已加入忽略规则，不进入运行证据、提交材料或模型上下文。

## 运行完整流程

### 1. 不调用模型的准备检查

```bash
.venv/bin/python -m pytest -q
.venv/bin/python run_demo.py --prepare-only --output .runs/prepare
```

成功准备返回退出码 **2**，表示“实际模型调用和修改后验收尚待完成”，不是整体通过。
程序会实际执行：原有回归测试、目标功能失败、seekdb 入库与检索、规格更新重导入、
PowerContext 状态写入，以及新进程重启后的状态恢复和上下文构造。

### 2. MiniMax 实际修改并验证

```bash
.venv/bin/python run_demo.py --generate --output .runs/live
```

每次使用新的输出目录，不覆盖既有运行记录。脚本复制原始工程，获取实时上下文，
调用 MiniMax，让模型仅修改 `tasks.py`，然后在另一个副本中应用模型返回的最终代码。
相同目标测试必须在修改前因行为断言失败、修改后全部通过，原有回归测试也必须全部通过。
测试文件哈希在前后保持一致。

退出码 **0** 表示本轮六项均通过，**1** 表示执行失败；详情见输出目录 `report.json`。
如果模型代码未通过，不声称完成，也不自动改写模型的代码或放宽测试。

### 3. 评审从干净状态复现

取得真实调用证据后，将该轮 `model/`、代码前后快照、补丁、测试日志和报告整理进提交材料。
以下命令会从初始样例重新建立数据库、重新运行前后测试，使用已记录的真实模型最终代码：

```bash
.venv/bin/python run_demo.py --replay evidence/live/model --output .runs/review
```

复现不需要作者的 Key，也不要求模型重新生成完全相同的补丁。模型请求记录、最终输出和代码
哈希会相互核对；新环境检索的当前源文件内容及恢复的任务状态必须与原始调用一致。
原始开发记录用于核对模型的实际参与，不能用手写或模拟的模型记录替代。

## 模块 CLI

```bash
python -m sdd_context --state-dir /tmp/sdd-manual --task sample/task.json ingest --project-root sample/project
python -m sdd_context --state-dir /tmp/sdd-manual --task sample/task.json context
python -m sdd_context --state-dir /tmp/sdd-manual --task sample/task.json state
python -m sdd_context --state-dir /tmp/sdd-manual --task sample/task.json save --input progress.json
```

`progress.json` 示例：`{"done":["已确认边界缺陷"],"todo":["修改实现并测试"],"validation":{"target_exit_code":1}}`。
`context` 输出当前规格、代码、测试的路径、行号、摘要、实际正文，以及 PowerContext 精确引用、
恢复的进度和后续待办。查询阶段只读 seekdb，不以源目录文件绕过数据库。

修改规格后再次 `ingest`，项目索引原子替换。下次 `context` 即使用新版本；删除的工程文件也不再返回。
可以更换任务描述和检索词；模块不读取样例的 `required_sources` 作为检索答案。
该字段只供验收核对必要来源。

## 六项要求与检查

| 官方要求 | 本项目验证方式 | 证据 |
|---|---|---|
| 能够运行 | 新目录初始化，完整 CLI/模型/测试流程 | `report.json`、运行命令 |
| MiniMax 实际参与 | 实际请求包含任务和检索上下文；最终输出与修改文件一致 | `model/request.json`、`model/record.json`、`change.patch` |
| 上下文正确 | 必要规格/代码/测试来源真实；规格 revision 2 重导入后替换 revision 1 | `initial-sources.json`、`context-after-restart.json` |
| 数据库实际参与 | 实际 seekdb 表保存资料；参数化 SQL 的返回正文用于模型输入 | `initial-import.json`、`updated-import.json`、`database-queries.json` |
| 状态能够复用 | 空状态写入真实 PowerContext；新进程读取，并成为后续上下文和待办 | `state-write.json`、`context-after-restart.json` |
| 代码修改有效 | 目标测试前失败后通过；回归通过；测试哈希相同 | 前后 `*-target.*`、`*-regression.*`、代码快照与报告 |

## 范围与限制

本例采用小规模工程文件的 SQL 词项匹配，按规格、代码、测试分组检索，不宣称实现了任意代码库的语义理解。
只导入 `.py` 和 `.md`，不导入密钥、二进制或完整个人仓库。示例项目、规格和测试均为本作品原创。
外部依赖遵守其自身许可证；PowerContext 和 seekdb 的技术来源分别为官方开源项目。
没有将 UI、扩展能力或额外测试数量当成得分条件。

题面：[MiniMax：规有所依](https://oceanbase.yuque.com/org-wiki-obtech-vh7w9r/hc3pio/qlqnoryzrs7gtqbw)。
提交分支必须基于官方 `2026-competition-ecosystem`，本项目资产位于 `minimax/`。
