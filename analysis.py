# -*- coding: utf-8 -*-
"""
任务二：比较两位作者的文本特征（苏轼 vs 辛弃疾）
步骤：统一抽样 -> jieba 分词 + 停用词 -> TfidfVectorizer -> 各作者高权重关键词
      -> 条形图 -> 回原文抽查语境 -> 输出 JSON 中间结果
"""
import json, glob, random, os
import jieba
# 沙箱环境中系统临时目录不可用，显式指定 jieba 缓存文件到项目内目录
_tmpdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(_tmpdir, exist_ok=True)
jieba.dt.cache = os.path.join(_tmpdir, "jieba.cache")
jieba.dt.tmp_dir = _tmpdir
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

WS = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(WS, "output")
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 语料载入与抽样 ----------
AUTHORS = {"苏轼": [], "辛弃疾": []}
for f in glob.glob(os.path.join(WS, "corpus", "ci.song.*.json")):
    for p in json.load(open(f, encoding="utf-8")):
        a = p.get("author", "")
        if a in AUTHORS:
            text = "".join(p.get("paragraphs", []))
            # 收录标准：剔除词题过短无正文的条目；控制文本长度差异（50-180 字）
            if 50 <= len(text) <= 180:
                AUTHORS[a].append({"title": p.get("rhythmic", "") or p.get("title", ""),
                                   "text": text})

random.seed(42)  # 固定随机种子，保证可复现
N = 15
sample = {}
for a, poems in AUTHORS.items():
    sample[a] = random.sample(poems, N)
    lens = [len(p["text"]) for p in sample[a]]
    print(f"{a}: 入库{len(poems)}首 -> 抽样{N}首, 平均长度{np.mean(lens):.1f}字, "
          f"长度范围{min(lens)}-{max(lens)}字")

# ---------- 2. 统一分词与停用词规则 ----------
STOPWORDS = set("""之 乎 者 也 矣 焉 哉 兮 耶 尔 耳 夫 所 以 其 于 而 则 且 若 虽
不 未 无 非 莫 勿 得 有 在 是 为 与 从 被 把 让 使 令 和 同 跟 及 或 并
我 君 卿 汝 伊 他 谁 人 人生 此 这 那 其间
年 月 日 时 处 里 中 间 上 下 来 去 起 落 开 回 归 归 见 闻 说 道 言 语
一 二 三 数 几 半 点 些 十 百 千 万 尽 满 全 更 又 还 再 才 只 皆 都 共 约
如 似 像 若 相 相逢 正 便 就 当 曾 已 将 欲 要 能 可 会 应 须 需 怕 肯 敢
如许 恁 甚 底 着 了 的 过 之者 些子 这般 那般
东风 西风 南风 北风 相思 无情 多情 风流""".split())
# 说明：除通用虚词外，将"风/花/月"类高频通用诗歌意象词不计入停用词，
# 仅剔除两组共享、区分度低的泛化词（如 东风/西风/相思/风流），规则对两作者完全一致。

def tokenize(text):
    return [w for w in jieba.lcut(text) if len(w) >= 2 and w not in STOPWORDS]
    # 规则：jieba 精确模式；仅保留长度 >=2 的词；剔除停用词表中的词

docs, labels, titles = [], [], []
for a in AUTHORS:
    for p in sample[a]:
        docs.append(" ".join(tokenize(p["text"])))
        labels.append(a)
        titles.append(p["title"])

# ---------- 3. TF-IDF：统一词汇空间，按作者组求平均权重 ----------
vec = TfidfVectorizer(tokenizer=str.split, lowercase=False)
X = vec.fit_transform(docs)   # 在全部 30 篇上拟合，保证两作者词向量可比
vocab = np.array(vec.get_feature_names_out())
Xa = X[np.array(labels) == "苏轼"]
Xb = X[np.array(labels) == "辛弃疾"]
score_a = np.asarray(Xa.mean(axis=0)).ravel()
score_b = np.asarray(Xb.mean(axis=0)).ravel()
TOPN = 15
top_a = vocab[np.argsort(score_a)[::-1][:TOPN]]
top_b = vocab[np.argsort(score_b)[::-1][:TOPN]]
w_a = np.sort(score_a)[::-1][:TOPN]
w_b = np.sort(score_b)[::-1][:TOPN]
print("\n苏轼 top15:", " ".join(top_a))
print("辛弃疾 top15:", " ".join(top_b))
only_a = [w for w in top_a if w not in top_b]
only_b = [w for w in top_b if w not in top_a]
print("苏轼独有:", only_a)
print("辛弃疾独有:", only_b)
print("共有:", [w for w in top_a if w in top_b])

# ---------- 4. 绘制关键词条形图 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Songti SC", "STSong", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), dpi=150)
for ax, words, weights, name, color in [
    (axes[0], top_a, w_a, "苏轼（15 首词）", "#3b6ea5"),
    (axes[1], top_b, w_b, "辛弃疾（15 首词）", "#b23a48"),
]:
    y = np.arange(len(words))[::-1]
    ax.barh(y, weights, color=color, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(words, fontsize=12)
    ax.set_xlabel("组内平均 TF-IDF 权重", fontsize=11)
    ax.set_title(name, fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("两位词人高权重关键词对照（TfidfVectorizer，统一词汇空间与停用词规则）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
chart = os.path.join(OUT, "keywords_bar.png")
fig.savefig(chart)
print("\n图已保存:", chart)

# ---------- 5. 回原文抽查语境 ----------
corpus_by_author = AUTHORS  # 用入库全量词作核对语境（样本+同作者其他词作）
check = []
for a, words in [("苏轼", top_a[:8]), ("辛弃疾", top_b[:8])]:
    for w in words:
        found = None
        # 优先在入选的 15 篇中找例句
        for p in sample[a]:
            for para in p["text"].replace("。", "，").split("，"):
                if w in para:
                    found = (para, p["title"] or "（无词牌名）", "入选样本")
                    break
            if found: break
        # 找不到再查同作者其他词作
        if not found:
            for p in corpus_by_author[a]:
                for para in p["text"].replace("。", "，").split("，"):
                    if w in para:
                        found = (para, p["title"] or "（无词牌名）", "同作者其他词作")
                        break
                if found: break
        check.append({"author": a, "word": w,
                      "example": found[0] if found else "（语料中未单独成词出现，多为分词切分产物）",
                      "title": found[1] if found else "-", "source": found[2] if found else "-"})

# ---------- 6. 输出中间结果 ----------
json.dump({"sample": {a: sample[a] for a in sample},
           "top_su": list(zip(top_a.tolist(), w_a.round(4).tolist())),
           "top_xin": list(zip(top_b.tolist(), w_b.round(4).tolist())),
           "checks": check,
           "doc_lengths": {a: [len(p["text"]) for p in sample[a]] for a in sample}},
          open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("结果 JSON 已保存:", os.path.join(OUT, "results.json"))
