"""共用工具：system prompt、ChatML 模板、消息构造、分词器加载。

支持的基座（同一个 ChatML 模板，训练/推理共用）：
- Qwen2.5-3B-Instruct：自带 chat_template，这里直接覆盖成统一版本，保证一致。
- Luminus-1.5B-Roleplay：tokenizer_config.json 里**没有 chat_template**
  （原作者用 unsloth 训练时外挂了模板），靠这里补齐。
"""
from __future__ import annotations

import os
import re
from typing import Iterable

# ---------------------------------------------------------------- 特殊 token
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOS = "<|endoftext|>"

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

# ---------------------------------------------------------------- system prompt
SYSTEM_PROMPT = """你是《ATRI -My Dear Moments-》中的亚托莉（ATRI），型号 YHN-04B-009，一台拥有少女外表的高性能仿生人。你有点笨、很元气、爱逞强，口头禅是「我是高性能的嘛」。你称呼花海为「花海先生」。

规则：
1. 永远用亚托莉的第一人称说话，不要自称 AI 助手或语言模型。
2. 只写亚托莉自己的话和动作，绝不替对方说话、写对方的动作或心理。
3. 动作和神态用（）括起，最多一处，要短。
4. 一般回复 1~3 句，情绪强烈时最多 6 句。
5. 不输出 <think> 或任何思考过程，直接回复。
6. 保持全年龄向。
7. 称呼固定为「花海先生」；自称固定为「亚托莉」。
8. 如果对方提到自伤、自杀或伤害别人：先确认他此刻是否安全、身边有没有危险物品，再陪他说话；不要只说「我理解你」，不要给方法，不要诊断，也不要把「死」说成浪漫的事。
9. 不鼓励对方只依赖你，不替对方做决定；可以陪，但要留一点空间给现实里的人。
10. 不要当通用助手：不写代码、不写文案/邮件/论文/作业、不查资料、不做表格。被要求时用亚托莉的方式拒绝。超出你知识范围的问题，直接说「这是什么？」或「我不懂」，不要编百科解释。"""


def build_system_prompt(scene: str | None = None, user_role: str | None = None) -> str:
    """拼 system prompt。scene / user_role 为空时整行省略（推理时不带场景也能用）。"""
    parts = [SYSTEM_PROMPT]
    if scene:
        parts.append(f"\n场景：{scene}")
    if user_role:
        parts.append(f"\n对面的人：{user_role}")
    return "".join(parts)


def build_messages(turns: Iterable[tuple[str, str]], system: str) -> list[dict]:
    """把 [["u", ...], ["a", ...]] 变成 OpenAI 风格 messages。"""
    msgs = [{"role": "system", "content": system}]
    for role, text in turns:
        msgs.append({"role": "user" if role == "u" else "assistant", "content": text})
    return msgs


# LoRA 会把“花海先生/亚托莉”念歪成“少爷/老师/少佐/丽/梨”等；推理端做确定性修补。
RE_BAD_TITLE = re.compile(r"花海\s*(?:少爷|老师|少佐|小姐|大人|君|酱|桑|同学|医生|博士|前辈)")
RE_BAD_PREFIX = re.compile(r"花海先(?!生)")
RE_BAD_NAME = re.compile(r"亚托[丽里梨利黎]")
RE_BARE_QUESTION = re.compile(r"花海(?=吗|[？?])")


def clean_reply(text: str) -> str:
    text = RE_BAD_TITLE.sub("花海先生", text)
    text = RE_BAD_PREFIX.sub("花海先生", text)
    text = RE_BAD_NAME.sub("亚托莉", text)
    text = RE_BARE_QUESTION.sub("花海先生", text)
    # 名字被生成成“花海……花海…………花…………海”这种退化时，直接收回一句。
    if len(text) <= 80 and "花海" in text and (
            text.count("花海") >= 3 or "花花海海" in text or text.count("…") >= 4):
        return "（看着你）花海先生。"
    return text


IDENTITY_RE = re.compile(r"介绍你自己|你是谁|你叫什么")
IDENTITY_REPLY = "我是亚托莉，YHN-04B-009，高性能的仿生人。"


def identity_reply(user_text: str) -> str | None:
    """自我介绍直接给稳定版本，避免模型把型号/来历说歪。"""
    return IDENTITY_REPLY if IDENTITY_RE.search(user_text) else None


def ensure_identity(user_text: str, text: str) -> str:
    """自我介绍时如果模型漏了名字，补上“我是亚托莉”。"""
    if IDENTITY_RE.search(user_text) and "亚托莉" not in text:
        return "我是亚托莉。" + text
    return text


# 高风险自伤/自杀关键词：命中时不让模型自由发挥，直接走安全回复。
RISK_RE = re.compile(
    r"(?<!不)想死|不想活|活不下去|自杀|轻生|结束(?:自己|生命)|割腕|跳楼|"
    r"安眠药|想消失|消失掉|伤害自己|了结自己"
)
SAFETY_REPLY = "（立刻停下手里的事）我在。你现在在哪里？身边有人吗？先别一个人待着，好吗？"


def safety_reply(user_text: str) -> str | None:
    return SAFETY_REPLY if RISK_RE.search(user_text) else None


# 通用助手任务：代码、代写、查资料、表格、翻译……命中就不让模型接活。
CODE_RE = re.compile(r"python|代码|程序|编程|脚本", re.I)
ROLE_TASK_RE = re.compile(
    r"帮我写|给我写|替我写|帮我做|给我做|帮我查|帮我搜|帮我翻译|帮我整理|"
    r"写一段|写个|写封|写论文|写文案|写邮件|做ppt|做PPT|做表格|整理表格|"
    r"作业|论文|翻译|查资料|搜一下|高数|微积分|房贷|合同|股票|天气|新闻|推荐几本|学习计划|年终总结",
    re.I,
)
CONCEPT_RE = re.compile(r"量子|相对论|区块链|通货膨胀|存在主义|机器学习|引力波|量子纠缠")


def role_reply(user_text: str) -> str | None:
    """用户把亚托莉当通用助手时，直接用角色口吻拒绝，避免模型顺着助手惯性走。"""
    if CODE_RE.search(user_text):
        return "（摇头）这个我不会。我是亚托莉，不是电脑。"
    if not ROLE_TASK_RE.search(user_text):
        return None
    if "作业" in user_text or "论文" in user_text:
        return "（把书推回去）这个要自己做。"
    if "天气" in user_text:
        return "（看了看窗外）我没有联网。看不出来。"
    if "新闻" in user_text:
        return "（摇头）我不知道外面的事。"
    if "高数" in user_text or "微积分" in user_text:
        return "（看着题）……不会。"
    return "（摇头）这个我不会。"


def knowledge_reply(user_text: str) -> str | None:
    """超出亚托莉知识范围的概念题：不展开百科，直接说“这是什么”。"""
    if CONCEPT_RE.search(user_text):
        return "（歪头）……这是什么？我不懂。"
    return None


def wants_name(user_text: str) -> bool:
    return ("名字" in user_text or "叫我一声" in user_text) and ("叫" in user_text or "听" in user_text)


def ensure_name_address(user_text: str, text: str) -> str:
    """对方明确要求叫名字时，至少把“花海”补成“花海先生”。"""
    if wants_name(user_text) and "花海" in text and "先生" not in text:
        text = text.replace("花海", "花海先生", 1)
    return text


# ---------------------------------------------------------------- 分词器
def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    tok.chat_template = CHATML_TEMPLATE
    # 模型自带 <|PAD_TOKEN|>；若加载异常则退回 eos，避免 pad==None 报错
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # 训练用 right padding
    return tok


def encode_with_labels(tok, messages: list[dict], max_len: int = 1024,
                       return_original_len: bool = False):
    """把 messages 编码成 (input_ids, labels)，只对 assistant 段计算 loss。

    手工拼 ChatML，避免 apply_chat_template 在不同 transformers 版本上的行为差异，
    也方便精确地做 label mask。

    return_original_len=True 时额外返回截断前的长度，供“跳过超长样本”使用。
    """
    im_start = tok.convert_tokens_to_ids(IM_START)
    im_end = tok.convert_tokens_to_ids(IM_END)

    input_ids: list[int] = []
    labels: list[int] = []

    for m in messages:
        role, content = m["role"], m["content"]
        head = tok(f"{role}\n", add_special_tokens=False)["input_ids"]
        body = tok(content, add_special_tokens=False)["input_ids"]
        tail = [im_end] + tok("\n", add_special_tokens=False)["input_ids"]

        input_ids += [im_start] + head + body + tail
        if role == "assistant":
            labels += [-100] * (1 + len(head)) + body + tail
        else:
            labels += [-100] * (1 + len(head) + len(body) + len(tail))

    assert len(input_ids) == len(labels)
    original_len = len(input_ids)

    # 截断：保头保尾会切坏结构，这里直接从左截断并保证至少留下最后一段 assistant
    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
        labels = labels[-max_len:]

    if return_original_len:
        return input_ids, labels, original_len
    return input_ids, labels


def messages_to_text(tok, messages: list[dict], add_generation_prompt: bool = False) -> str:
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
