"""【07 / 08】评分决策与最终汇总，无模型或数据库调用。

align 解决“一个用户表述对应哪一个库内名称”；aggregate 解决“整个问题是否都确认”。
例如用户同时问 A、B：先分别 align，再 aggregate；不能拿 A 的 0.95 淘汰 B 的 0.78。
阅读顺序：normalize_name/model_conflict → align → aggregate。
下方分数例子是人为构造的分支演示，不代表实际模型一定返回这些分数。
"""

from decimal import Decimal
import re
import unicodedata


def normalize_name(name: str) -> str:
    """只统一字符和空白，保留型号中的数字、标点及后缀。

    例："  ＲＳ-12   数字万用表 " → "rs-12 数字万用表"。
    NFKC 统一兼容字符，casefold 统一大小写，split/join 收敛连续空白。
    不删除型号连字符；"L420" 与 "L420x" 仍然是不同名称。
    """
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


# 对模型输出的常见泛称再做程序侧兜底；这不是覆盖所有品类的商品词典。
GENERIC_NAMES = {"万用表", "数字万用表", "电压表", "电脑", "笔记本", "笔记本电脑",
                 "路由器", "网关", "产品", "商品", "设备", "电池", "说明书", "用户手册"}


def model_tokens(name: str) -> set[str]:
    """提取明显的型号词元："RS PRO RS-12 数字万用表" → {"rs-12"}。"""
    # 仅保守识别同时含拉丁字母和数字的型号；不声称能理解所有品牌命名规则。
    tokens = re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", normalize_name(name))
    return {t for t in tokens if re.search(r"[a-z]", t) and re.search(r"[0-9]", t)}


def model_conflict(source: str, target: str) -> bool:
    """两边都有可识别型号时，要求用户提到的型号包含在目标名称里。

    "L420" → "华为 L420x" 为冲突；"RS-12" → "RS PRO RS-12" 不冲突。
    某侧没有可识别型号时返回 False，表示本规则无法发现冲突，不表示已证明身份相同。
    """
    expected, actual = model_tokens(source), model_tokens(target)
    return bool(expected and actual and not expected.issubset(actual))


class ItemNameAligner:
    """阈值、候选去重、产品决策和最终输出集中在此，可用固定分数独立测试。"""
    def __init__(self, config):
        self.config = config

    def align(self, mention, search, mention_id: str) -> dict:
        """【07】将一次检索的文档级命中，变为一个表述的产品级决策。

        输入：mention.name="RS-12"，search.hits 为对应名称的混合检索结果。
        返回：{mention_id: "m1", status: "confirmed", confirmed: 商品组, candidates: [...], ...}。
        status 也可能是 needs_clarification 或 not_found；这是单个表述的状态。
        """
        # 【07.1】先按名称分组，再比较第一、第二名。
        # 例：d1/d2 都是“RS-12 数字万用表”，分数 0.85/0.84，只形成一个分数 0.85 的商品组。
        groups = {}
        for hit in search.hits:
            key = normalize_name(hit.item_name)
            if key not in groups:
                groups[key] = dict(item_name=hit.item_name, score=hit.score, matched_pks=[],
                                   matched_document_ids=[], file_titles=[])
            group = groups[key]
            # 多份文档只提供来源，不增加该商品的置信度。
            group["score"] = max(group["score"], hit.score)
            for field, value in (("matched_pks", hit.pk), ("matched_document_ids", hit.document_id),
                                 ("file_titles", hit.file_title)):
                if value not in group[field]:
                    group[field].append(value)
        # 【07.2】默认 <0.45 丢弃，>=0.45 保留为候选；0.7 此时还没有参与确认。
        # max(score) 不等于累加分数，同商品说明书多并不意味着它更符合用户的问题。
        candidates = sorted((g for g in groups.values() if g["score"] >= self.config.item_name_mid_confidence),
                            key=lambda g: (-g["score"], normalize_name(g["item_name"])))
        result = {"mention_id": mention_id, **mention.model_dump(), "status": "not_found",
                  "reason": "no_match", "confirmed": None, "candidates": candidates,
                  "warnings": list(search.warnings)}
        if not candidates:
            # 情况 A：没有命中，或最高分仅 0.44 → 这个表述 not_found。
            return result
        result.update(status="needs_clarification", reason="below_confirmation_threshold")
        if search.truncated:
            # 情况 B：候选被 TopK 截断 → 尚不能判断是否存在同样合适的其他商品。
            # 即使当前看到 0.99 的结果，也保持澄清，不能假设它一定唯一。
            result["reason"] = "recall_truncated"
            return result
        high = [g for g in candidates if g["score"] >= self.config.item_name_high_confidence]
        if not high:
            # 情况 C：最高分是 0.62 → 保留候选，但没有任何商品跨过 0.7 确认门槛。
            return result
        # 【07.3】只有高分结果才能走精确名称确认；相等不是字符串“包含”关系。
        # 例：提取“RS-12”，库内只有“RS-12 数字万用表”，并不走 exact，而走后续候选判断。
        exact = [g for g in high if normalize_name(g["item_name"]) == normalize_name(mention.name)]
        picked = exact[0] if len(exact) == 1 else candidates[0]
        if model_conflict(mention.name, picked["item_name"]):
            # 情况 D：用户问 L420，命中 L420x，即使 0.9 也不能用相似度掩盖型号差异。
            result["reason"] = "model_conflict"
            return result
        if exact:
            # 情况 E：完整规范名称精确一致，且已满足 >=0.7 → 确认该商品。
            reason = "exact_name"
        elif len(candidates) == 1:
            # 情况 F：没有精确名称，但只有一个 >=0.45 的候选且它 >=0.7 → 确认。
            # 不是“只有一个 >=0.7 就确认”；0.701、0.699 是两个候选，必须检查分差。
            reason = "unique_candidate"
        else:
            # 【07.4】仅比较同一表述下两个不同商品，不与其他用户表述的分数比较。
            # Decimal(str(...)) 稳定处理 0.90-0.75 的十进制边界，不给分数偷偷加 epsilon。
            gap = Decimal(str(candidates[0]["score"])) - Decimal(str(candidates[1]["score"]))
            if gap < Decimal(str(self.config.item_name_score_gap)):
                # 情况 G：0.82 与 0.80 差 0.02 <0.15 → 保留候选，请用户澄清。
                result["reason"] = "close_candidates"
                return result
            reason = "score_gap"
            # 情况 H：0.90 与 0.75 差恰好 0.15 → 第一名满足领先要求，可确认。
        result.update(status="confirmed", reason=reason, confirmed=picked)
        return result

    def aggregate(self, results: list[dict], original_query: str) -> dict:
        """【08】将所有 align 结果汇总成节点最终输出，供基类返回到图。

        全部 confirmed → item_names 发布给下游；全部 not_found → 无匹配提示。
        没有表述、存在歧义、或部分确认 → needs_clarification，item_names 仍为空。
        confirmed_items 可保留部分成功供展示，不能把它当作下游已获准检索的范围。
        """
        from knowledge.processor.query_processor.state import empty_result
        result = empty_result()
        # 【08.1】重新构建本次输出，清空旧 answer、options、error_code；明细原样保留。
        result["mention_results"] = results
        result["warnings"] = list(dict.fromkeys(w for r in results for w in r["warnings"]))
        # dict.fromkeys 按首次出现顺序去重，比 set 更适合稳定展示诊断和文档来源。
        confirmed = {}
        for row in results:
            if row["confirmed"]:
                item = row["confirmed"]
                key = normalize_name(item["item_name"])
                if key not in confirmed:
                    # 复制组内列表，避免汇总追加来源时修改 results 中的单表述明细。
                    confirmed[key] = {**item, "matched_pks": list(item["matched_pks"]),
                                      "matched_document_ids": list(item["matched_document_ids"]),
                                      "file_titles": list(item["file_titles"])}
                else:
                    previous = confirmed[key]
                    previous["score"] = max(previous["score"], item["score"])
                    for field in ("matched_pks", "matched_document_ids", "file_titles"):
                        previous[field] = list(dict.fromkeys(previous[field] + item[field]))
        result["confirmed_items"] = list(confirmed.values())
        unresolved = [r for r in results if r["status"] != "confirmed"]
        if results and not unresolved:
            # 【08.2 成功】A=0.95、B=0.78 且分别已确认 → A/B 都保留，不全局淘汰 B。
            # 例：item_names=["RS-12 数字万用表"]，answer=""，error_code=""。
            result.update(item_confirm_status="confirmed", item_names=[c["item_name"] for c in confirmed.values()])
            bindings = "；".join(f"{r['name']} → {r['confirmed']['item_name']}" for r in results)
            # 绑定独立商品范围，原问题全文保留，避免 LLM 改写丢掉否定或比较条件。
            result["rewritten_query"] = f"商品名称对应关系：{bindings}。用户问题：{original_query}"
            return result
        if results and all(r["status"] == "not_found" for r in results):
            # 【08.3 无匹配】有提取目标但全都没找到；与下面“没有提取到产品”分开表达。
            result.update(item_confirm_status="not_found", answer="未找到匹配的产品，请核对商品名称或型号。")
            return result
        result["item_confirm_status"] = "needs_clarification"
        if not results:
            # 【08.4 缺产品】如“这个怎么用？”没有可靠上下文，提示补充名称或型号。
            result["answer"] = "请提供具体商品名称或型号，以便查询对应的产品资料。"
            return result
        # 【08.5 歧义/部分确认】轮流分配展示名额，完整候选保留在 mention_results。
        # 例：A/B 各有 4 个候选、总额度 5 → A1、B1、A2、B2、A3，避免 B 完全看不到选项。
        selected = {r["mention_id"]: [] for r in unresolved}
        remaining = self.config.item_name_max_options
        for index in range(self.config.item_name_max_options):
            for row in unresolved:
                if remaining and index < len(row["candidates"]):
                    selected[row["mention_id"]].append(row["candidates"][index])
                    remaining -= 1
        prompts = []
        for row in unresolved:
            # 每组保留 mention_id 与原表述，以便界面知道用户正在确认哪个目标。
            # has_more=True 可能是展示名额不够，也可能是数据库召回达到上限。
            choices = selected[row["mention_id"]]
            result["options"].append(dict(mention_id=row["mention_id"], extracted_name=row["name"],
                                          candidates=choices, has_more=len(choices) < len(row["candidates"])
                                          or "candidate_recall_truncated" in row["warnings"]))
            if choices:
                prompts.append(f"“{row['name']}”是指 {'、'.join(c['item_name'] for c in choices)} 中的哪一款？")
            else:
                prompts.append(f"请补充“{row['name']}”的准确名称或型号。")
        # 【08.6】部分成功示例：“已确认：RS-12。请补充‘电脑’的准确名称或型号。”
        # 此时仍不写 item_names/rewritten_query，避免只回答了问题的一部分便继续检索。
        prefix = "已确认：" + "、".join(c["item_name"] for c in confirmed.values()) + "。" if confirmed else ""
        result["answer"] = prefix + " ".join(prompts)
        return result
