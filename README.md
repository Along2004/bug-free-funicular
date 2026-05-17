# bitsandbytes     ：4-bit 量化載入大型語言模型（降低 GPU 記憶體需求）
# rank_bm25        ：BM25 關鍵字搜尋演算法
# sentence-transformers：多語言向量模型（語意搜尋用）
# jieba            ：中文斷詞工具
# accelerate       ：HuggingFace 加速套件（量化模型必需）

!pip install -U bitsandbytes rank_bm25 sentence-transformers jieba accelerate pymupdf tools
!pip uninstall fitz -y

# 套件用途一覽：
#   json                   - 讀寫 JSON/JSONL 格式檔案
#   torch                  - PyTorch 深度學習框架（模型運算基礎）
#   typing.List            - 型別提示，讓程式碼更易讀
#   transformers           - HuggingFace 模型庫，用來載入 Gemma LLM
#     AutoModelForCausalLM   自動載入因果語言模型（文字生成用）
#     AutoTokenizer          自動載入對應的斷詞器
#     BitsAndBytesConfig     設定 4-bit 量化參數
#     pipeline               包裝模型推論的簡易介面
#   tqdm                   - 顯示進度條（適用 Jupyter Notebook）
#   jieba                  - 中文斷詞工具（把中文句子切成詞彙列表）
#   rank_bm25.BM25Okapi    - BM25 關鍵字搜尋模型
#   sentence_transformers  - 把文字轉成向量，用於語意相似度計算

import json
import torch
from typing import List
from transformers import (
    AutoModelForCausalLM,           # 自動載入因果語言模型
    AutoTokenizer,                  # 自動載入對應的 tokenizer
    BitsAndBytesConfig,             # 設定量化模型的參數
    pipeline                        # 簡易模型推論介面
)
from tqdm.autonotebook import tqdm  # 顯示進度條（適用於 Jupyter Notebook）
import jieba                        # 中文斷詞工具
from rank_bm25 import BM25Okapi     # BM25 搜尋模型
from sentence_transformers import SentenceTransformer  # 向量模型
import fitz

# ── 定義常見問題─────────────────────────
data_public = [
    {"query": "旅平險包含「海外看醫生」嗎？"},
    {"query": "刷信用卡送的旅平險夠用嗎？"},
    {"query": "遇到班機延誤或取消，怎麼賠？"},
]

print(f"── 共 {len(data_public)} 筆範例問題 ──")
for i, item in enumerate(data_public):
    print(f"{i+1:3d}. {item['query']}")

# ── 套用5家旅平險條款的資料夾────────

import pymupdf
import os

folder_path = "/content/drive/MyDrive/旅平險條文"

fulltext = ""

for filename in os.listdir(folder_path):

    if filename.endswith(".pdf"):

        file_path = os.path.join(folder_path, filename)

        doc = fitz.open(file_path)

        for page in doc:
            fulltext += page.get_text()

        print(f"{filename} 讀取完成")

print(f"\n總字元數：{len(fulltext):,}")

# ── 取出 Query 字串 ─────────────────────────────────────────
# data_public 是一個 list，每個元素是一個 dict，格式如下：
#   {"query": "問題字串"}
#
# 這裡用 list comprehension 遍歷 data_public，
# 從每個 dict 中取出 'query' 這個 key 對應的字串，
# 組成一個純字串的 list，存入 queries。
#
# 等同於以下迴圈寫法：
#   queries = []
#   for item in data_public:
#       queries.append(item['query'])
queries = [item['query'] for item in data_public]

# 印出所有 query，確認資料正確載入
print(f"── 共 {len(queries)} 個問題 ──")
for i, q in enumerate(queries):
    print(f"{i+1}. {q}")

chunks = []
passage = ""
chunk_size = 512                # 每段文字的長度上限
stride = 256                    # 每段間的步長（TODO: Boss baseline 請調整此值）

folder_path = "/content/drive/MyDrive/旅平險條文"

for filename in os.listdir(folder_path):

    if filename.endswith(".pdf"):

        file_path = os.path.join(folder_path, filename)

        doc = fitz.open(file_path)

        for page in doc:
          passage += page.get_text()

        print(f"{filename}讀取完成")

print(f"\n總字元數:{len(passage)}")

length = len(passage)
for i in range(0, length, stride):
    chunk = passage[i:i+chunk_size]  # 依照 chunk_size 切出文字段落
    if len(chunk) > 0:
      chunks.append(chunk)

# ── 印出第一個 chunk ──
print(f"共切出 {len(chunks)} 個 chunks")
print("=== 第一個 chunk ===")
print(chunks[0])

# --- BM25 初始化 ---
tokenized_chunks = [list(jieba.cut(chunk)) for chunk in chunks]  # 對每個段落斷詞
bm25 = BM25Okapi(tokenized_chunks)  # 建立 BM25 索引

# --- 印出 BM25 結果（部分）---
print("═" * 55)
print("  [BM25] 斷詞結果預覽")
print("═" * 55)
print(f"共 {len(tokenized_chunks)} 個 tokenized chunk")
print(f"\nChunk #0 原文（前 40 字）：")
print(f"  {chunks[0][:40]}...")
print(f"\nChunk #0 斷詞（前 15 個 token）：")
print(f"  {tokenized_chunks[0][:15]}")
print(f"\nChunk #0 token 總數：{len(tokenized_chunks[0])}")
print("═" * 55)

# --- Embedding 模型初始化 ---
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")  # 載入向量模型
chunks_embeddings = embedding_model.encode(chunks, show_progress_bar=True)  # 預先計算所有段落的向量

# --- 印出 Embedding 結果（部分）---
print("═" * 55)
print("  [Embedding] 向量結果預覽")
print("═" * 55)
print(f"向量矩陣維度：{chunks_embeddings.shape}")
print(f"  → {chunks_embeddings.shape[0]} 個 chunk，每個 chunk 為 {chunks_embeddings.shape[1]} 維向量")
print(f"\nChunk #0 原文（前 40 字）：")
print(f"  {chunks[0][:40]}...")
print(f"\nChunk #0 向量（前 8 維）：")
print(f"  {chunks_embeddings[0][:8]}")
print("═" * 55)
print()
print("【對比小結】")
print(f"  BM25     → Chunk #0 表示為 {len(tokenized_chunks[0])} 個離散 token")
print(f"  Embedding→ Chunk #0 表示為 {chunks_embeddings.shape[1]} 維連續浮點向量")

# ── 函式一：BM25 檢索 ────────────────────────────────────
# 輸入：查詢字串、所有段落、BM25 模型
# 輸出：依相關性由高到低排序的段落列表
# 流程：問題斷詞 → 計算每段落分數 → 排序 → 回傳
def bm25_retrieve(query: str, chunks: List[str], bm25: BM25Okapi) -> List[str]:
    tokenized_query = list(jieba.cut(query))          # 對查詢斷詞
    scores = bm25.get_scores(tokenized_query)         # 計算各段落的 BM25 分數
    rank = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)  # 由高到低排序
    return [chunk for chunk, _ in rank]               # 只回傳段落文字（不含分數）

# ── 函式二：Embedding 向量檢索 ───────────────────────────
# 輸入：查詢字串、所有段落、預計算的段落向量、向量模型
# 輸出：依語意相似度由高到低排序的段落列表
# 流程：問題向量化 → 與所有段落計算餘弦相似度 → 排序 → 回傳
def embedding_retrieve(query: str, chunks: List[str], chunks_embeddings: List[str], embedding_model: SentenceTransformer) -> List[str]:
    query_embeddings = embedding_model.encode(query, prompt_name="query")              # 把問題轉成向量
    scores = list(embedding_model.similarity(query_embeddings, chunks_embeddings)[0])  # 計算與每個段落的相似度
    rank = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)              # 由高到低排序
    return [chunk for chunk, _ in rank]

# ── 函式三：RRF 融合排序（Reciprocal Rank Fusion）────────
# 用途：把多個排序列表合併成一個最終排序
# 原理：每個段落的最終分數 = Σ 1/(k + 在各排名中的名次)
#       兩種方法都認為相關的段落，分數最高
# 範例：某段落在 BM25 排名第2、向量排名第3（k=60）
#       分數 = 1/(60+2) + 1/(60+3) = 0.0161 + 0.0159 = 0.0320
# 優點：不需手動調整權重，自動平衡兩種方法的優缺點
def reciprocal_rank_fusion(*ranked_lists, k=60) -> List[str]:
    scores = {}
    for rl in ranked_lists:
        for rank, doc_id in enumerate(rl, start=1):  # rank 從 1 開始計算
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))  # 依分數由高到低排序
    return [d for d, _ in fused]

# ╔══════════════════════════════════════════════════════════╗
# ║                    載入語言模型（LLM）                    ║
# ╚══════════════════════════════════════════════════════════╝
#
# 使用的模型：Gemma-3-4b-it（Google 開源，40億參數，instruction-tuned）
#
# 【4-bit 量化說明】
#   模型原始精度是 32-bit float，4B 參數 ≈ 16GB 記憶體
#   量化後壓縮成 4-bit，記憶體降至 ~4GB，精度損失極小
#   nf4（NormalFloat4）：專為 AI 模型設計的 4-bit 格式，效果優於一般 int4
#   double quant：對量化參數再做一次量化，進一步節省 ~0.4GB
#   bfloat16：計算時使用 16-bit 浮點，在精度與速度間取得平衡
#
# 【pipeline 參數說明】
#   max_new_tokens=256 : 模型最多生成 256 個 token（約 200 個中文字）
#   do_sample=False    : 關閉隨機性，使用貪婪解碼（每次選最高機率的字）
#                        → 相同輸入永遠得到相同輸出，結果穩定可重現

# --- 設定 4-bit 量化參數 ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # 啟用 4-bit 量化載入
    bnb_4bit_use_double_quant=True,        # 啟用雙重量化（額外節省記憶體）
    bnb_4bit_quant_type="nf4",             # 使用 NormalFloat4 格式
    bnb_4bit_compute_dtype=torch.bfloat16  # 計算時使用 bfloat16
)

# --- 載入語言模型 ---
llm = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="unsloth/gemma-3-4b-it",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,  # 減少 CPU 記憶體暫用量
)

# --- 載入 Tokenizer ---
# Tokenizer 負責把文字轉成模型看得懂的數字（token ids），以及反向轉回文字
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it")

# --- 建立推論 Pipeline ---
# pipeline 是 HuggingFace 提供的高階介面，把 tokenize → 推論 → decode 包成一步
llm_pipe = pipeline(
    "text-generation",
    model=llm,
    tokenizer=tokenizer,
    max_new_tokens=256,  # 回應最多 256 個 token
    do_sample=False,     # 貪婪解碼，結果確定可重現
    temperature=None,
    top_p=None,
    top_k=None,
)

# 本程式定義了三個 prompt：
#   SIMPLE_SYSTEM_PROMPT  用於 Simple Baseline（無 RAG）
#   SYSTEM_PROMPT         用於有 RAG 的 Baseline
#   USER_PROMPT           有 RAG 時的使用者輸入模板

# 為什麼加 .strip()？ ──────────────────────────────────────
# 三引號字串開頭和結尾會多出換行字元。
# .strip() 把首尾的空白與換行去掉，避免模型收到多餘的空行干擾。

SIMPLE_SYSTEM_PROMPT = """
你是一位精通旅遊平安保險的專家。
請根據提問，直接提供精簡並準確的解答，避免與問題無關的內容。
""".strip()

SYSTEM_PROMPT = """
你是一位精通旅遊平安保險的專家。
請根據**問題**，從使用者提供的**旅平險文章片段**找出與問題相關的內容，直接提供精簡並準確的解答，避免與問題無關的內容。
""".strip()

# 為什麼用 {query} / {relevant_chunks} 佔位符？ ────────────
# USER_PROMPT 是模板字串，執行時用 Python 的 .format() 動態填入：
#   user_input = USER_PROMPT.format(query=query, relevant_chunks=relevant_chunks)
# 這樣可以把「格式」和「資料」分開，每個 query 都套同一個模板。

USER_PROMPT = """
### 問題
{query}
### 旅平險文章片段
{relevant_chunks}
""".strip()

responses = []
for query in tqdm(queries, desc="Response Generation"):  # ① 逐一取出每個 query
    chats = [                                             # ② 組裝對話格式
        {'role': "system", 'content': SIMPLE_SYSTEM_PROMPT},  # 角色設定，不含任何資料
        {'role': "user",   'content': query},                  # 直接放問題
    ]
    response = llm_pipe(chats)[0]['generated_text'][-1]['content'].strip()  # ③ 推論，取最後一輪回覆
    responses.append(response[:512])                      # ④ 截斷至 512 字元後存入列表

# ── Input & Output Examples ──────────────────────────────────
for i, (query, resp) in enumerate(zip(queries, responses)):
    print(f"\n{'='*60}")
    print(f"Example {i+1}")
    print(f"{'─'*60}")
    print(f"[Input]")
    print(f"  system : {SIMPLE_SYSTEM_PROMPT}")
    print(f"  user   : {query}")
    print(f"{'─'*60}")
    print(f"[Output]")
    print(f"  model  : {resp}")

responses = []
inputs = []
for query in tqdm(queries, desc="Response Generation"):  # ① 逐一取出每個 query
    # ② BM25 檢索：問題斷詞 → 計算分數 → 排序 → 取前 10 段 → 用空行串接
    relevant_chunks = "\n\n".join(bm25_retrieve(query=query, chunks=chunks, bm25=bm25)[:10])
    # ③ 填入 USER_PROMPT 佔位符，組成「問題 + 10 段相關原文」的完整使用者訊息
    user_input = USER_PROMPT.format(query=query, relevant_chunks=relevant_chunks)
    chats = [                                             # ④ 組裝對話格式
        {'role': "system", 'content': SYSTEM_PROMPT},        # 角色設定（有 RAG 版本）
        {'role': "user",   'content': user_input},            # 問題 + 10 段相關原文
    ]
    response = llm_pipe(chats)[0]['generated_text'][-1]['content'].strip()  # ⑤ 推論，取最後一輪回覆
    responses.append(response[:512])                      # ⑥ 截斷至 512 字元後存入列表
    inputs.append(user_input)

# ── Input & Output Examples ──────────────────────────────────
for i, (user_input, resp) in enumerate(zip(inputs, responses)):
    print(f"\n{'='*60}")
    print(f"Example {i+1}")
    print(f"{'─'*60}")
    print(f"[Input]")
    print(f"  system : {SYSTEM_PROMPT}")
    print(f"  user   :\n{user_input}")
    print(f"{'─'*60}")
    print(f"[Output]")
    print(f"  model  : {resp}")
