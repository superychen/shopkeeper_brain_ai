# shopkeeper_brain_ai
知识库问答系统。

`knowledge/` 是独立的 Python 子项目；其 `.env`、`pyproject.toml`、`uv.lock`、
`requirements.txt` 和虚拟环境均放在该目录中。

```powershell
# 进入知识库 Python 子项目
Set-Location .\\knowledge

# Windows PowerShell：启用子项目虚拟环境
.\\.venv\\Scripts\\Activate.ps1

# 项目固定使用 Python 3.12；首次创建或重建环境
uv sync --python 3.12

# 新增依赖并同步现有环境
uv add <package>
uv sync
```

主要源码位于 `knowledge/`：`api/`、`processor/`、`prompt/`、`service/`、
`schema/` 和 `test/` 已预先创建。

PDF/Markdown 导入流水线位于 `knowledge/processor/import_processor/`，现在完整串接：

```text
文件入口 → PDF转Markdown（仅PDF） → 图片描述与MinIO上传
→ 文档切分 → DeepSeek商品名识别及商品向量入库
→ BGE-M3切片双向量编码 → Milvus幂等写入与读回核验
```

在项目根目录执行导入（将文件路径替换为实际路径）：

```powershell
& '.\knowledge\.venv\Scripts\python.exe' -m knowledge.processor.import_processor 'D:\资料\说明书.pdf'
# 同一业务文档内容更新时使用稳定标识，成功后自动清理多余旧切片。
& '.\knowledge\.venv\Scripts\python.exe' -m knowledge.processor.import_processor 'D:\资料\说明书.md' --document-id 'manual-001'
```

配置读取 `knowledge/.env`，完整模板见 `knowledge/.env.example`。切片集合默认
`kb_chunks_v1`，商品名称集合默认 `kb_item_names_v1`。DeepSeek负责名称抽取和已有
视觉接口的图片描述；本地BGE-M3生成1024维dense和sparse向量。`FlagEmbedding`
已纳入依赖锁文件，首次加载会下载模型至当前HuggingFace缓存。

成功结果输出文档标识、写入数量、集合名和质量告警；失败返回非零退出码。图片描述
降级或部分上传失败会保留告警。无唯一商品名时仍可导入正文。修改模型或连接配置后
请重启进程；不同模型版本应使用新集合，不能只因维度相同就混写。

程序调用入口为 `run_import_graph(state, config)`。第一版使用单进程导入锁，
多批次及商品/切片集合之间没有文档级事务，导入期间应暂停该文档的在线查询。
`build_import_graph`用于低层编排与测试，调用方自行保证串行；目前未配置持久化
checkpointer，故障恢复方式是重新提交完整导入。

测试从项目根目录运行：

```powershell
& '.\knowledge\.venv\Scripts\python.exe' -m unittest discover -s knowledge/test -p 'test_*.py'
# 真实数据库测试：创建唯一命名的测试集合，并在结束时清理。
$env:RUN_MILVUS_INTEGRATION='1'
& '.\knowledge\.venv\Scripts\python.exe' -m unittest knowledge.test.test_milvus_integration -v
# 完整模型验收会调用DeepSeek，并实际加载BGE-M3与MinerU。
$env:RUN_FULL_IMPORT_INTEGRATION='1'
& '.\knowledge\.venv\Scripts\python.exe' -m unittest knowledge.test.test_full_import_integration -v
```

完整设计、实施记录与验收范围见 `docs/知识库导入全流程与切片向量化入库技术设计报告.md`。

查询第一步“商品名确认”已提供独立节点与最小 LangGraph：DeepSeek 提取商品表述，
BGE-M3 生成查询 dense/sparse，Milvus 使用 COSINE/IP 混合检索及等权 WeightedRanker。
融合分数至少 0.7 且无歧义才自动确认；多商品问题全部确认后才发布查询范围。

```powershell
& '.\knowledge\.venv\Scripts\python.exe' -m knowledge.processor.query_processor 'RS PRO RS-12 数字万用表怎么测电阻？'
& '.\knowledge\.venv\Scripts\python.exe' -m unittest knowledge.test.test_item_name_confirm -v
```

本阶段返回商品范围、候选澄清或错误状态，尚不执行正文检索和答案生成。
配置、程序入口及真实服务验收见 `docs/商品名确认节点开发与联调说明.md`。
