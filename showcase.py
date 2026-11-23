"""本机 MovieLens 展示页。用项目 Python 运行本文件，再访问控制台地址。"""
import csv
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8765
MODES = {"single": "单智能体", "multi_serial": "多智能体串行", "multi_parallel": "多智能体并行"}


def h(value):
    return escape(str(value), quote=True)


def replay(index):
    with (ROOT / "outputs/w7_repeat_r06/repeat_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not 0 <= index < len(rows):
        raise ValueError("未知回放记录")
    row = rows[index]
    options = "".join(
        f'<option value="{i}" {"selected" if i == index else ""}>第 {h(r["repeat"])} 轮 · {h(MODES[r["mode"]])} · '
        f'{"通过" if r["overall_pass"] == "True" else "未通过"}</option>'
        for i, r in enumerate(rows))
    selector = f'<form method="get"><label for="record">R06 实验记录</label><select id="record" name="record">{options}</select> <button>查看记录</button></form>'
    passed = row["overall_pass"] == "True"
    body = f'<h2>{h(row["question"])}</h2>' + status(row["hard_check"], row["judge_verdict"], row["elapsed_seconds"])
    body += f'<h3>历史回答</h3><pre>{h(row["answer"]) if passed else "本轮未通过，已隐藏候选回答。"}</pre>'
    body += f'<details><summary>查看评审依据</summary><p>成功调用的专家　{h(row["successful_routes"])}</p><p>{h(row["judge_comments"])}</p><small>来源　outputs/w7_repeat_r06/repeat_results.csv</small></details>'
    return selector + body


def status(hard, verdict, seconds):
    return (f'<dl><div><dt>数字核验</dt><dd>{h(hard)}</dd></div><div><dt>审查结果</dt><dd>{h(verdict)}</dd></div>'
            f'<div><dt>执行耗时</dt><dd>{float(seconds):.2f} s</dd></div></dl>')


def live(question):
    from multi_agent import run
    state = run(question)
    body = f'<h2>{h(question)}</h2>' + status("PASS" if state["hard_check"] == "PASS" else "FAIL",
                                               state["verdict"], state["elapsed_seconds"])
    body += f'<h3>最终回答</h3><pre>{h(state["final"])}</pre>'
    if state["verdict"] == "FAIL" and state["hard_check"].startswith("FAIL:"):
        reason = state.get("block_reason", "")
        if "无证据数字" in reason:
            body += "<p>核验详情：草稿包含原始查询或文档未提供的数值。</p>"
        elif "错配" in reason:
            body += "<p>核验详情：有数值与对应对象或指标不一致。</p>"
    body += "<h3>工具取证记录</h3>"
    for record in state["tool_records"]:
        evidence = record["text"] if record["status"] in {"success", "empty"} else "未取得有效证据"
        label = {"success": "取证成功", "empty": "查询完成，结果为空", "error": "取证失败"}
        body += f'<details><summary>{h(record["tool_name"])} · {h(label.get(record["status"], record["status"]))}</summary><pre>{h(evidence)}</pre></details>'
    return body


def page(content, online=False):
    layout = (ROOT / "showcase.html").read_text(encoding="utf-8")
    return (layout.replace("{{replay_current}}", "" if online else 'aria-current="page"')
            .replace("{{live_current}}", 'aria-current="page"' if online else "")
            .replace("{{content}}", content)).encode("utf-8")


FORM = """<form method="post" action="/ask" onsubmit="this.querySelector('button').disabled=true;this.querySelector('button').textContent='分析中…'">
<label for="question">向 MovieLens 提问</label><textarea id="question" name="question" required maxlength="1200" placeholder="例如：轻度和重度用户分别有多少人？"></textarea>
<button>提交分析 ↗</button><small> 使用项目配置的模型服务</small></form>"""


class Handler(BaseHTTPRequestHandler):
    def show(self, content, online=False, code=200):
        data = page(content, online)
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/ask":
            return self.show(FORM, True)
        if url.path != "/":
            return self.send_error(404)
        try:
            index = int(parse_qs(url.query).get("record", ["1"])[0])
            self.show(replay(index))
        except (ValueError, OSError):
            self.show("<p>回放记录不可用，请检查实验文件或返回首页。</p>", code=404)

    def do_POST(self):
        if self.path != "/ask":
            return self.send_error(404)
        if self.headers.get("Origin") != f"http://{HOST}:{PORT}":
            return self.send_error(403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 20000:
                return self.send_error(400)
            question = parse_qs(self.rfile.read(length).decode()).get("question", [""])[0].strip()
            if not 0 < len(question) <= 1200:
                return self.show(FORM + "<p>请输入 1 至 1200 个字符。</p>", True, 400)
            self.show(FORM + live(question), True)
        except Exception:
            self.show(FORM + "<p role='alert'>实时分析未完成。请检查项目模型连接，或打开实验回放。</p>", True, 503)


if __name__ == "__main__":
    print(f"http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()

