"""离线验收矩阵：阻断输出、工具状态、Fact 关系与声明式格式。"""
import asyncio
from types import SimpleNamespace
import unittest

import multi_agent as ma
from trust_layer import audit_answer, source_status


class TrustLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cluster_result = ma.cluster_profiles_result()
        cls.cluster_record = cls.cluster_result.to_record()
        cls.cluster_evidence = cls.cluster_result.text

    def test_sql_statuses_and_display_metadata(self):
        success = ma.run_sql_result(
            "SELECT m.title, COUNT(*) AS n, AVG(r.rating) AS avg_rating "
            "FROM ratings r JOIN movies m ON m.movieId = r.movieId "
            "GROUP BY m.title ORDER BY n DESC LIMIT 20"
        )
        self.assertEqual(success.status, "success")
        self.assertEqual(success.row_count, 20)
        self.assertEqual(success.display_count, 15)
        self.assertIn("完整查询结果共 20 行", success.text)
        self.assertIn("当前展示 15 行", success.text)
        self.assertIn("排序依据：n DESC", success.text)

        empty = ma.run_sql_result("SELECT * FROM ratings WHERE 1 = 0")
        self.assertEqual(empty.status, "empty")
        error = ma.run_sql_result("SELECT FROM")
        self.assertEqual(error.status, "error")
        self.assertNotIn(error.text, ma.usable_raw_evidences([error.to_record()]))
        self.assertEqual(source_status([], "sql"), "not_called")

    def test_cluster_subject_swap_is_rejected(self):
        ok = audit_answer("轻度用户 532 人，重度用户 78 人。",
                          self.cluster_evidence, [self.cluster_record])
        self.assertTrue(ok["ok"], ok)

        swapped = audit_answer("轻度用户 78 人，重度用户 532 人。",
                               self.cluster_evidence, [self.cluster_record])
        self.assertFalse(swapped["ok"])
        self.assertTrue(any("错配" in item for item in swapped["errors"]))

    def test_fact_reference_subject_and_metric_binding(self):
        records = [self.cluster_record]
        # F003 是轻度用户 user_count（F001/F002 是 K 与轮廓系数）。
        ok = audit_answer("轻度用户数量为 [F003] 人。", self.cluster_evidence, records)
        self.assertTrue(ok["ok"], ok)
        self.assertEqual(ok["rendered_answer"], "轻度用户数量为 532 人。")

        wrong_subject = audit_answer("重度用户数量为 [F003] 人。", self.cluster_evidence, records)
        self.assertFalse(wrong_subject["ok"])
        self.assertTrue(any("F003" in item and "错配" in item for item in wrong_subject["errors"]))

    def test_sign_and_declared_transformations(self):
        signed = [{
            "tool_name": "mock", "status": "success",
            "facts": [{"subject": "指标A", "metric": "delta", "value": 3,
                       "display": "+3", "unit": "score", "aliases": ["指标A"]}],
        }]
        self.assertTrue(audit_answer("指标A +3", "指标A +3", signed)["ok"])
        self.assertFalse(audit_answer("指标A -3", "指标A +3", signed)["ok"])

        ratio = [{
            "tool_name": "mock", "status": "success",
            "facts": [{"subject": "指标A", "metric": "score", "value": 0.7757,
                       "display": "0.7757", "unit": "ratio", "aliases": ["指标A"]}],
        }]
        evidence = "指标A 0.7757"
        self.assertFalse(audit_answer("指标A为0.78", evidence, ratio)["ok"])
        self.assertTrue(audit_answer("指标A保留两位小数为0.78", evidence, ratio)["ok"])
        self.assertTrue(audit_answer("指标A为[F001:round:2]", evidence, ratio)["ok"])

        percent = [{
            "tool_name": "mock", "status": "success",
            "facts": [{"subject": "指标A", "metric": "user_ratio", "value": 0.872,
                       "display": "0.872", "unit": "ratio", "aliases": ["指标A"]}],
        }]
        self.assertTrue(audit_answer("指标A占比为87.2%", "指标A 0.872", percent)["ok"])

        already_percent = [{
            "tool_name": "mock", "status": "success",
            "facts": [{"subject": "4.0分占比", "metric": "pct", "value": 26.6,
                       "display": "26.6", "unit": "percent", "aliases": ["4.0分占比"]}],
        }]
        double = audit_answer("4.0分占比 [F001:percent:1]。", "4.0分 26.6", already_percent)
        self.assertFalse(double["ok"])
        self.assertTrue(any("已是百分数" in item for item in double["errors"]))
        # 已百分化的 Fact 直接引用渲染原值，视为合法。
        self.assertTrue(audit_answer("4.0分占比 [F001]。", "4.0分 26.6", already_percent)["ok"])

    def test_duplicate_number_next_to_fact_ref(self):
        records = [{
            "tool_name": "mock", "status": "success",
            "facts": [{"subject": "轻度用户", "metric": "user_count", "value": 532,
                       "unit": "人", "aliases": ["轻度用户"]}],
        }]
        audit = audit_answer("轻度用户 [F001] 532 人。", "轻度用户 532", records)
        self.assertFalse(audit["ok"])
        self.assertTrue(any("重复书写" in item for item in audit["errors"]))
        # 正常引用（不重复写数值）不受影响。
        self.assertTrue(audit_answer("轻度用户 [F001] 人。", "轻度用户 532", records)["ok"])

    def test_hard_failure_does_not_show_draft(self):
        state = {
            "routes": ["mining"], "tool_records": [self.cluster_record],
            "tool_evidences": {"mining": self.cluster_evidence},
            "draft": "内部草稿：共有 999 人。", "question": "用户有多少人？",
            "revision_count": 1,
        }
        result = asyncio.run(ma.critic(state))
        self.assertEqual(result["verdict"], "FAIL")
        self.assertNotIn("999", result["final"])
        self.assertIn("程序硬校验未通过", result["final"])
        self.assertIn("999", result["block_reason"])

    def test_semantic_failure_does_not_show_draft(self):
        class RejectLLM:
            async def ainvoke(self, messages):
                return SimpleNamespace(content='{"verdict":"REVISE","comments":"结论与证据不一致"}')

        original_llm = ma.llm
        ma.llm = RejectLLM()
        try:
            state = {
                "routes": ["mining"], "tool_records": [self.cluster_record],
                "tool_evidences": {"mining": self.cluster_evidence},
                "draft": "内部草稿：轻度用户 532 人。", "question": "轻度用户有多少人？",
                "revision_count": 1,
            }
            result = asyncio.run(ma.critic(state))
        finally:
            ma.llm = original_llm
        self.assertEqual(result["verdict"], "FAIL")
        self.assertNotIn("轻度用户 532 人", result["final"])
        self.assertIn("审查未通过", result["final"])
        self.assertIn("结论与证据不一致", result["block_reason"])

    def test_parenthetical_context_binds_inside_subject(self):
        """括号注释内的 Fact 属于括号内的对象，不受括号后实体干扰（R10 回归）。"""
        records = [{
            "tool_name": "mock", "status": "success",
            "facts": [
                {"subject": "K-Means", "metric": "silhouette", "value": 0.4077,
                 "display": "0.4077", "unit": "系数", "aliases": ["K-Means", "聚类"]},
                {"subject": "轻度用户", "metric": "user_count", "value": 532,
                 "unit": "人", "aliases": ["轻度用户", "轻度"]},
            ],
        }]
        draft = "分群（K-Means，K=2，轮廓系数 [F001:round:2]）：轻度用户 [F002] 人。"
        self.assertTrue(audit_answer(draft, "K-Means K=2 轮廓系数 0.4077；轻度用户 532",
                                     records)["ok"])

    def test_list_marker_and_compound_label_numbers_ignored(self):
        """行首列表编号与 Top-10 连字符标签不算事实数字（R02/R03/R04 回归）。"""
        records = [{
            "tool_name": "mock", "status": "success",
            "facts": [{"subject": "Top10查询", "metric": "row_count", "value": 10,
                       "unit": "行", "aliases": ["Top10查询"]}],
        }]
        draft = ("评价人数最多的 10 部电影：\n\n"
                 "10. Schindler's List (1993)：[F001] 人\n"
                 "表中仅展示 Top-10 规则。")
        evidence = "评价人数最多的 10 部电影；Schindler's List (1993) 197 人"
        audit = audit_answer(draft, evidence, records)
        self.assertTrue(audit["ok"], msg=str(audit["errors"]) + str(audit["missing_numbers"]))

    def test_short_alias_in_generic_phrase_does_not_steal_binding(self):
        """「分为轻度与重度两类」里的短别名「轻度」不抢绑 K-Means 的数字（R04 回归）。"""
        records = [{
            "tool_name": "mock", "status": "success",
            "facts": [
                {"subject": "K-Means", "metric": "best_k", "value": 2,
                 "unit": "类", "aliases": ["K-Means"]},
                {"subject": "K-Means", "metric": "silhouette", "value": 0.4077,
                 "display": "0.4077", "unit": "系数", "aliases": ["K-Means"]},
                {"subject": "轻度用户", "metric": "user_count", "value": 532,
                 "unit": "人", "aliases": ["轻度用户", "轻度"]},
            ],
        }]
        draft = ("K-Means 最优 K 为 [F001]，轮廓系数 [F002:round:2]，"
                 "分为轻度与重度两类。轻度用户 [F003] 人。")
        audit = audit_answer(draft, "K-Means K=2 轮廓系数 0.4077；轻度用户 532", records)
        self.assertTrue(audit["ok"], msg=str(audit["errors"]) + str(audit["missing_numbers"]))


if __name__ == "__main__":
    unittest.main()
