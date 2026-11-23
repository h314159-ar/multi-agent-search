"""可信回答基础组件：工具状态、结构化事实与声明式数字校验。

本模块不调用 LLM，便于离线回归。Fact ID 只在内部草稿中使用，最终答案由程序渲染。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Any


@dataclass
class ToolResult:
    """工具的内部结构化结果；text 仅供 LLM 阅读，error 不进入事实证据。"""

    tool_name: str
    status: str  # success / empty / error
    text: str
    args: dict[str, Any] = field(default_factory=dict)
    facts: list[dict[str, Any]] = field(default_factory=list)
    row_count: int | None = None
    display_count: int | None = None
    sort_by: str = ""
    error: str = ""

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?[%％]?")
FACT_REF_RE = re.compile(r"\[(F\d{3})(?::(round|percent):(\d+))?\]")

METRIC_ALIASES = {
    "user_count": ("用户数量", "用户数", "人数"),
    "user_ratio": ("用户占比", "用户比例", "占比", "比例"),
    "avg_rating": ("平均评分", "评分均值", "均分"),
    "avg_ratings": ("平均评分数", "人均评分", "评分数", "评分数量", "条评分"),
    "avg_active_years": ("平均活跃年限", "活跃年限", "活跃"),
    "rating_count": ("评价人数", "评分数", "评分数量"),
    "row_count": ("结果总数", "总行数", "命中总数"),
    "silhouette": ("轮廓系数",),
    "best_k": ("最佳 K", "最佳K", "K=", "K＝"),
    "support": ("支持度", "support"),
    "confidence": ("置信度", "confidence"),
    "lift": ("提升度", "lift"),
}

CN_DIGITS = {"零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def clean_text(text: str) -> str:
    return (text or "").replace("**", "").replace("％", "%")


def _decimal(token: str) -> Decimal:
    token = token.replace(",", "").replace("％", "%")
    if token.endswith("%"):
        token = token[:-1]
    return Decimal(token)


def _key(token: str) -> str:
    """数值归一化；正号与无符号等价，负号保留，百分比不与原值混同。"""
    percent = token.replace("％", "%").endswith("%")
    value = _decimal(token).normalize()
    return f"{value}%" if percent else str(value)


def extract_numbers(text: str) -> list[str]:
    """提取事实数字；忽略行首有序列表编号与「Top-10」这类连字符标签数字。"""
    values = []
    for line in _strip_list_markers(clean_text(text)).splitlines():
        for match in NUMBER_RE.finditer(line):
            start = match.start()
            if (start >= 2 and line[start - 1] in "-–—"
                    and line[start - 2].isascii() and line[start - 2].isalpha()):
                continue
            values.append(match.group().replace(",", "").replace("％", "%"))
    return values


def hard_validate_numbers(draft: str, evidence_text: str) -> tuple[bool, list[str], int]:
    """兼容旧调用：只检查数字出处，不声称它能判断对象关系。"""
    draft_numbers = extract_numbers(draft)
    evidence_numbers = {_key(n) for n in extract_numbers(evidence_text)}
    missing = sorted({n for n in draft_numbers if _key(n) not in evidence_numbers})
    return not missing, missing, len(draft_numbers)


def format_decimal(value: Any, decimals: int | None = None, percent: bool = False) -> str:
    number = Decimal(str(value))
    if percent:
        number *= 100
    if decimals is not None:
        quantum = Decimal(1).scaleb(-decimals)
        number = number.quantize(quantum, rounding=ROUND_HALF_UP)
        return f"{number:.{decimals}f}" + ("%" if percent else "")
    text = format(number.normalize(), "f")
    return text + ("%" if percent else "")


def assign_fact_ids(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按原始工具记录生成 Fact 目录；失败工具中的内容不得进入目录。"""
    catalog: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("status") not in ("success", "empty"):
            continue
        for source_fact in record.get("facts", []):
            fact = dict(source_fact)
            fact_id = f"F{len(catalog) + 1:03d}"
            fact["fact_id"] = fact_id
            fact.setdefault("display", format_decimal(fact["value"]))
            fact.setdefault("aliases", [fact.get("subject", "")])
            fact["source_tool"] = record.get("tool_name", "")
            catalog[fact_id] = fact
    return catalog


def fact_catalog_text(records: list[dict[str, Any]]) -> str:
    catalog = assign_fact_ids(records)
    if not catalog:
        return "（本次没有结构化 Fact；其他事实只能按原始证据原文引用）"
    lines = []
    for fact_id, fact in catalog.items():
        lines.append(
            f"[{fact_id}] 对象={fact.get('subject', '')}；指标={fact.get('metric', '')}；"
            f"原值={fact.get('display', fact['value'])}；单位={fact.get('unit', '')}；"
            f"来源={fact.get('source_tool', '')}:{fact.get('source_location', '')}"
        )
    return "\n".join(lines)


def render_fact_references(draft: str, records: list[dict[str, Any]]) -> tuple[str, list[str], set[str]]:
    """渲染 [F001]、[F001:round:2]、[F001:percent:1]，返回程序生成的数字集合。"""
    catalog = assign_fact_ids(records)
    errors: list[str] = []
    generated: set[str] = set()

    def replace(match: re.Match) -> str:
        fact_id, operation, digits_text = match.groups()
        fact = catalog.get(fact_id)
        if fact is None:
            errors.append(f"未知 Fact 引用：{fact_id}")
            return match.group(0)
        if operation:
            digits = int(digits_text)
            if not 0 <= digits <= 6:
                errors.append(f"{fact_id} 的小数位数超出 0~6：{digits}")
                return match.group(0)
            # 双重百分比防护：percent 渲染结果超过 100% 说明该 Fact 本身
            # 已是百分数（如 SQL 直接查出 26.60），禁止再乘 100（R13 回归）。
            if operation == "percent":
                try:
                    if fact.get("unit") != "ratio":
                        errors.append(f"{fact_id} 的值已是百分数，禁止叠加 percent 声明")
                        return match.group(0)
                except InvalidOperation:
                    pass
            rendered = format_decimal(fact["value"], digits, percent=(operation == "percent"))
        else:
            rendered = str(fact.get("display", format_decimal(fact["value"])))
        generated.update(extract_numbers(rendered))
        return rendered

    return FACT_REF_RE.sub(replace, clean_text(draft)), errors, generated


def _line_context(text: str, position: int) -> tuple[str, int]:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    if end < 0:
        end = len(text)
    return text[start:end], position - start


def _local_context(text: str, position: int) -> tuple[str, int]:
    """截取最近的语义段，避免跨分号/冒号把下一实体误配给当前数字。

    数字处在未闭合的括号对内时，边界收缩到该括号对内部：
    「（K-Means，轮廓系数 [F002]）：轻度用户」中 F002 属于 K-Means，
    而不是括号后的轻度用户；括号外的数字不受影响。
    """
    hard_delimiters = ("\n", "。", "；", ";", "：")
    openers = ("（", "(")
    closers = ("）", ")")
    start = max(text.rfind(d, 0, position) for d in hard_delimiters) + 1
    candidates = [text.find(d, position) for d in hard_delimiters
                  if d not in ("：",) and text.find(d, position) >= 0]
    segment = text[start:position]
    opens = sum(segment.count(o) for o in openers)
    closes = sum(segment.count(c) for c in closers)
    if opens > closes:
        start = max(text.rfind(o, start, position) for o in openers) + 1
        candidates += [text.find(c, position) for c in closers
                       if text.find(c, position) >= 0]
    end = min(candidates) if candidates else len(text)
    return text[start:end], position - start


def _subject_hits(line: str, catalog: dict[str, dict[str, Any]]) -> list[tuple[int, str, str]]:
    hits = []
    seen = set()
    for fact in catalog.values():
        subject = fact.get("subject", "")
        for alias in fact.get("aliases", [subject]):
            if not alias or len(alias) < 2:
                continue
            key = (alias, subject)
            if key in seen:
                continue
            seen.add(key)
            for match in re.finditer(re.escape(alias), line, re.I):
                hits.append((match.start(), subject, alias))
    return hits


def _nearest_subject_hit(line: str, position: int,
                         catalog: dict[str, dict[str, Any]]) -> tuple[int, str, str] | None:
    hits = _subject_hits(line, catalog)
    if not hits:
        return None
    # 距离按别名长度加权：短别名（如「轻度」）在「分为轻度与重度两类」
    # 这类泛指语境中不应抢绑数字，完整别名（「轻度用户」）优先。
    return min(hits, key=lambda hit: (abs(hit[0] - position) / max(len(hit[2]), 1),
                                      abs(hit[0] - position)))


def _nearest_subject(line: str, position: int, catalog: dict[str, dict[str, Any]]) -> str:
    hit = _nearest_subject_hit(line, position, catalog)
    return hit[1] if hit else ""


def _nearest_metric(line: str, position: int) -> str:
    hits = []
    for metric, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            for match in re.finditer(re.escape(alias), line, re.I):
                hits.append((match.start(), match.end(), metric, alias))
    if not hits:
        return ""
    # 先按别名终点与数字的距离，再按别名长度，确保“评价人数”优先于其中的“人数”。
    return min(hits, key=lambda hit: (abs(hit[1] - position), -len(hit[3])))[2]


def _validate_reference_relations(draft: str, catalog: dict[str, dict[str, Any]]) -> list[str]:
    errors = []
    for match in FACT_REF_RE.finditer(draft):
        fact_id = match.group(1)
        fact = catalog.get(fact_id)
        if not fact:
            continue
        line, position = _local_context(draft, match.start())
        subject_hit = _nearest_subject_hit(line, position, catalog)
        if subject_hit:
            subject_start, subject, alias = subject_hit
            if subject != fact.get("subject"):
                errors.append(f"{fact_id} 被错配给「{subject}」，实际属于「{fact.get('subject')}」")
            # 指标只看当前主体之后、当前数字之前的文字，避免跨后续指标误报。
            metric_start = subject_start + len(alias)
            metric_context = line[metric_start:position]
            metric = _nearest_metric(metric_context, len(metric_context))
            if metric and metric != fact.get("metric"):
                errors.append(f"{fact_id} 指标错配：语境是「{metric}」，实际是「{fact.get('metric')}」")
    return errors


def _strip_list_markers(text: str) -> str:
    """把行首有序列表编号（「10. 」「5、」）替换为等长空白，避免被当事实数字。"""
    def blank(match: re.Match) -> str:
        return " " * (match.end() - match.start())
    return re.sub(r"(?m)^\s*(?:\d+[、)]\s*|\d+\.\s+)", blank, text)


def _validate_direct_value_relations(rendered: str, catalog: dict[str, dict[str, Any]]) -> list[str]:
    """检查直接整数与唯一 Fact 的对象关系；小数关系交给 Fact 引用和格式声明。"""
    errors = []
    text = _strip_list_markers(clean_text(rendered))
    for match in NUMBER_RE.finditer(text):
        token = match.group()
        if token.endswith("%"):
            continue
        try:
            key = _key(token)
        except InvalidOperation:
            continue
        candidates = []
        for fact in catalog.values():
            try:
                if _key(str(fact.get("display", fact["value"]))) == key:
                    candidates.append(fact)
            except (InvalidOperation, TypeError):
                continue
        if len(candidates) != 1:
            continue
        fact = candidates[0]
        line, position = _local_context(text, match.start())
        # 优先采用数值前的主体，也支持“片名：评分”。
        prefix = re.split(r"[\n。；;]", text[:match.start()])[-1]
        prior_hits = _subject_hits(prefix, catalog)
        if prior_hits:
            subject_hit = max(prior_hits, key=lambda hit: (hit[0] + len(hit[2]), len(hit[2])))
            line, position = prefix, len(prefix)
        else:
            subject_hit = _nearest_subject_hit(line, position, catalog)
        if subject_hit:
            subject_start, subject, alias = subject_hit
            if subject != fact.get("subject"):
                errors.append(f"数字 {token} 被错配给「{subject}」")
            metric_context = line[subject_start + len(alias):position]
            metric = _nearest_metric(metric_context, len(metric_context))
            if metric and metric != fact.get("metric"):
                errors.append(f"数字 {token} 的指标与「{subject}·{metric}」不一致")
    return errors


def _duplicate_number_errors(rendered: str) -> list[str]:
    """抓「[F003] 532」这类渲染后变成「532 532」的重复书写。"""
    errors = []
    pattern = re.compile(
        r"(?<![\d.])([-+]?\d+(?:\.\d+)?)(%?)([\s,，、]{1,4})\1(?![\d.%])")
    for match in pattern.finditer(clean_text(rendered)):
        errors.append(f"数字 {match.group(1)} 重复书写，Fact 引用处无需再写数值")
    return errors


def _declared_decimals(text: str) -> set[int]:
    decimals = {int(x) for x in re.findall(r"保留\s*(\d+)\s*位小数", text)}
    for chinese in re.findall(r"保留\s*([一二两三四五六])\s*位小数", text):
        decimals.add(CN_DIGITS[chinese])
    return decimals


def _round_matches(value: Decimal, target: Decimal, decimals: set[int]) -> bool:
    for digits in decimals:
        quantum = Decimal(1).scaleb(-digits)
        if value.quantize(quantum, rounding=ROUND_HALF_UP) == target:
            return True
    return False


def audit_answer(answer: str, evidence_text: str, records: list[dict[str, Any]],
                 question: str = "") -> dict[str, Any]:
    """统一验收：数字出处、对象/指标关系、显式舍入或百分比声明。"""
    raw = clean_text(answer)
    catalog = assign_fact_ids(records)
    rendered, reference_errors, generated = render_fact_references(raw, records)
    errors = list(reference_errors)
    errors.extend(_validate_reference_relations(raw, catalog))
    errors.extend(_validate_direct_value_relations(rendered, catalog))
    errors.extend(_duplicate_number_errors(rendered))

    source_text = clean_text(evidence_text + "\n" + question)
    source_tokens = extract_numbers(source_text)
    source_keys = {_key(n) for n in source_tokens}
    source_values = []
    for token in source_tokens:
        if not token.endswith("%"):
            try:
                source_values.append(_decimal(token))
            except InvalidOperation:
                pass
    generated_keys = {_key(n) for n in generated}
    decimals = _declared_decimals(raw)
    percent_declared = bool(re.search(r"百分比|占比|百分号|percent", raw, re.I))
    answer_numbers = extract_numbers(rendered)
    missing = []
    for token in answer_numbers:
        key = _key(token)
        if key in source_keys or key in generated_keys:
            continue
        target = _decimal(token)
        if token.endswith("%") and percent_declared:
            if any(_round_matches(value * 100, target, {max(-target.as_tuple().exponent, 0)})
                   for value in (_decimal(str(f["value"])) for f in catalog.values()
                                 if f.get("unit") == "ratio")):
                continue
        elif decimals and any(_round_matches(value, target, decimals) for value in source_values):
            continue
        missing.append(token)

    errors = list(dict.fromkeys(errors))
    missing = sorted(set(missing))
    return {
        "ok": not errors and not missing,
        "missing_numbers": missing,
        "errors": errors,
        "checked_numbers": len(answer_numbers),
        "rendered_answer": rendered,
        "fact_count": len(catalog),
    }


def route_for_tool(tool_name: str) -> str:
    if tool_name == "run_sql":
        return "sql"
    if tool_name in {"cluster_profiles", "association_rules", "decision_tree_summary"}:
        return "mining"
    if tool_name in {"retrieve_knowledge", "rag_retrieve"}:
        return "rag"
    return ""


def invoked_routes(records: list[dict[str, Any]]) -> set[str]:
    return {route for record in records if (route := route_for_tool(record.get("tool_name", "")))}


def successful_routes(records: list[dict[str, Any]]) -> set[str]:
    """success/empty 都表示工具成功执行；error 与 not_called 不计入覆盖。"""
    return {route for record in records
            if record.get("status") in ("success", "empty")
            and (route := route_for_tool(record.get("tool_name", "")))}


def source_status(records: list[dict[str, Any]], route: str) -> str:
    selected = [record for record in records if route_for_tool(record.get("tool_name", "")) == route]
    if not selected:
        return "not_called"
    if any(record.get("status") == "success" for record in selected):
        return "success"
    if any(record.get("status") == "empty" for record in selected):
        return "empty"
    return "error"
