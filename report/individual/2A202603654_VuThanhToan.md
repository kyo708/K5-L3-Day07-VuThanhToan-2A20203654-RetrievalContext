# Báo cáo Day 7

> Vũ Thanh Toàn · 2A202603654 · Ngày chạy: 2026-11-14

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** Bộ luật Lao động số 45/2019/QH14, bản scan sao y số 15/SY-VPCTN (83 trang, sha256 `cf6d8438…b8bb1a4`, xem [data/manifest.json](../../data/manifest.json)). Đây là văn bản quy phạm pháp luật, không thuộc đối tượng bảo hộ quyền tác giả (khoản 2 Điều 15 Luật SHTT), nên được phép sao chép và xử lý. Text lấy từ OCR của Day 6. Index Day 6 có 2 chiến lược: **A** (cố định 500 ký tự, chồng 50, 452 chunk) và **B** (theo đoạn, tối đa 1000 ký tự, **343 chunk**). Mọi thí nghiệm Day 7 dùng **index B**.
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16 + pgvector 0.8.7 (Docker, `127.0.0.1:55433`), bảng `chunks_768`, cosine `<=>`, exact search. Embedding `intfloat/multilingual-e5-base`, **768 chiều**, đã normalize, prefix `query: ` / `passage: `. Code lấy nguyên từ repo Day 6, commit `21e8f16` ([src/day6/README.md](../../src/day6/README.md)).
- **Agent model, cách chạy và phân chia dev/test:** `openai/gpt-4o-mini` qua OpenRouter, temperature 0, tối đa 6 lượt gọi tool. Benchmark có 30 câu: **dev** 15 câu có đáp án và 3 câu thiếu evidence, **test** 9 câu có đáp án và 3 câu thiếu evidence. 20 câu kế thừa benchmark Day 6, 10 câu mới viết ở Day 7. Loại câu: trực tiếp, diễn đạt lại, điều kiện/ngoại lệ, metadata filter, thiếu bằng chứng ([data/benchmark/questions.json](../../data/benchmark/questions.json)).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** câu có đáp án ghi `gold` gồm trang, điều và các trích dẫn nguyên văn bắt buộc. Một chunk được tính là hit khi nằm đúng trang và chứa **đủ mọi trích dẫn** (sau khi chuẩn hóa). Câu thiếu evidence ghi `absence_reason` và `check_method`, tức là cách đã tìm trên toàn corpus. Ví dụ u03: tìm "từ xa", "internet" ra 0 kết quả. Kết luận thiếu evidence không dựa vào việc top-k không tìm thấy. Câu thiếu evidence không tính vào mẫu số retrieval metrics. Agent chỉ thấy câu hỏi và filter, không thấy gold.

## Retrieval và thử nghiệm

Retrieval chạy bằng [src/day7/eval_retrieval.py](../../src/day7/eval_retrieval.py) trên câu hỏi gốc, không viết lại query, không dùng LLM. Dùng cùng corpus, index B, cùng nhãn. Context budget tính bằng **ký tự**, gồm cả dòng nguồn. Chi phí retrieval là 0 vì model chạy trên máy (CPU).

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline `dense_k3` | Dense e5-base trên pgvector Day 6, top-3, filter theo câu hỏi, budget 3000 | Baseline Day 6 | dev | 11/15 (0.73) | 13/15 (0.87) | 13/15 (0.87) | 165 ms / $0 |
| Phương pháp 2 `bm25_k3` | BM25 âm tiết có dấu, top-3, budget 3000 | Câu luật lặp lại thuật ngữ, khớp từ khóa có lợi | dev | 8/15 (0.53) | 14/15 (0.93) | 14/15 (0.93) | 4 ms / $0 |
| Phương pháp 3 `hybrid_k3` | RRF(dense, BM25), k=60, 30 candidate mỗi nhánh, top-3, budget 3000 | Kết hợp ngữ nghĩa và từ khóa | dev | 12/15 (0.80) | 14/15 (0.93) | 14/15 (0.93) | 186 ms / $0 |
| Cải thiện 1 `rerank_k3` | Hybrid lấy 20 candidate → `BAAI/bge-reranker-v2-m3`, top-3, budget 3000 | Cross-encoder sửa thứ hạng top-1 | dev | **14/15 (0.93)** | 14/15 (0.93) | 14/15 (0.93) | **31.122 ms** / $0 |
| Cải thiện 2 `hybrid_k3_nb` | Hybrid top-3 + 1 chunk lân cận mỗi bên, budget 6000 | Điều khoản bị cắt sang chunk kế bên (q07) | dev | 12/15 (0.80) | 14/15 (0.93) | **15/15 (1.00)** | 819 ms / $0 |
| Cấu hình đã chọn `hybrid_k3_nb` | như trên | đã chốt trên dev | test | 9/9 (1.00) | 9/9 (1.00) | 9/9 (1.00) | 100 ms / $0 |
| Baseline `dense_k3` | như trên | | test | 8/9 (0.89) | 8/9 (0.89) | 8/9 (0.89) | 116 ms / $0 |

Các cấu hình phụ đã thử trên dev:

| Cấu hình | Hit@1 | Hit@3 | CER | Thời gian | Nhận xét |
|---|---|---|---|---|---|
| `bm25fold_k3`: BM25 bỏ dấu + bigram | 9/15 | 12/15 | 12/15 | 6 ms | Giả thuyết "chịu được lỗi dấu OCR" **không đúng**: bỏ dấu làm mất phân biệt từ, kém hơn BM25 có dấu |
| `hybrid_fold_k3` | 11/15 | 13/15 | 13/15 | 179 ms | Kém hơn `hybrid_k3`, cùng lý do |
| `dense_k5`: top-5, budget 5000 | 11/15 | 13/15 | 14/15 | 187 ms | Tăng k chỉ thêm 1 câu |
| `dense_k3_nb` | 11/15 | 13/15 | 14/15 | 197 ms | Neighbours có ích cho dense |
| `rerank_k3_nb` | 14/15 | 14/15 | 14/15 | 30.193 ms | Rerank vẫn trượt q07 |

Kết quả đầy đủ (score, rank, top-10 từng câu): [results/retrieval/dev/](../../results/retrieval/dev/) và [results/retrieval/test/](../../results/retrieval/test/).

**Lỗi dev 1: model không chèn `[chunk_id]` vào câu trả lời** (lỗi ở tầng agent)
- *Hiện tượng:* với prompt ban đầu, GPT-4o-mini chỉ trả lời được 2/15 câu có đáp án. 13 câu dừng với lý do `invalid_citations`.
- *Nguyên nhân:* agent loại một citation nếu nó không có trong context **hoặc** không xuất hiện dạng `[id]` trong `answer` ([src/day7/agent.py](../../src/day7/agent.py)). Mình thêm trường `rejected_answer` vào trace để xem câu trả lời bị loại: cả 13 câu có nội dung đúng, mảng `citations` đúng, nhưng phần `answer` không có `[id]` nào. Quy tắc 4 của prompt chỉ nói "kèm citation" mà không nói phải đặt ở đâu.
- *Cách thử:* viết lại quy tắc 4: bắt buộc chèn `[chunk_id]` trong `answer` sau mỗi khẳng định, mỗi id một cặp ngoặc riêng, nói rõ hậu quả và đưa ví dụ. Mô tả trường `answer` trong schema tool cũng nhắc lại yêu cầu này. Giữ nguyên cấu hình (hybrid, top-3, nb=1, budget 6000) và luật kiểm tra.
- *Kết quả:* grounded TB từ 0.27 lên **1.93**, từ chối nhầm từ 13/15 xuống **0/15** (bảng ở phần Agent).

**Lỗi dev 2: q07, điều khoản bị cắt sang hai chunk** (lỗi ở tầng retrieval/context)
- *Hiện tượng:* q07 ("bị quấy rối tình dục có phải báo trước không?") trượt ở **mọi** cấu hình top-3, kể cả rerank.
- *Nguyên nhân:* evidence nằm ở hai chunk liền nhau. `p014:001` chứa "đơn phương chấm dứt hợp đồng lao động **không cần báo trước** trong trường hợp sau đây", còn `p014:002` chứa "d) Bị quấy rối tình dục tại nơi làm việc". Chunker B cắt giữa câu dẫn và danh sách.
- *Cách thử:* thêm 1 chunk lân cận mỗi bên cho mỗi hit (`neighbours=1`) và tăng budget lên 6000.
- *Kết quả:* CER của hybrid từ 14/15 lên **15/15**. Ở tầng agent: với nb=0, model vẫn trả lời "không cần báo trước" dù `p014:001` không có trong context, tức là dùng kiến thức ngoài tài liệu, chấm 0. Với nb=1, evidence đã có trong context nhưng model **vẫn chỉ trích `p014:002`**. Neighbours sửa được phần context, không sửa được citation.

**Lý do chọn cấu hình (chốt trên dev trước khi chạy test):** chọn `hybrid_k3_nb` vì cấu hình này có **CER cao nhất trên dev (15/15)** và grounded answer của agent cao nhất (1.93/2). Không chọn rerank: Hit@1 cao hơn nhưng không tăng CER, trong khi chậm khoảng **31 giây/câu** trên CPU, chậm hơn hybrid hơn 150 lần. Cái giá của lựa chọn này là context trung bình 4.716 ký tự so với 2.228 của baseline, và tốn thêm khoảng 70% prompt token ở agent.

**Chấm context 0/1/2 trên test** (9 câu có đáp án, cả hai cấu hình, [results/retrieval/test/context_reviews.json](../../results/retrieval/test/context_reviews.json)):

| Cấu hình | 0 | 1 | 2 | Ghi chú |
|---|---|---|---|---|
| Baseline `dense_k3` | 1 (q17) | 0 | 8 | q17: context chỉ có `p083:002` (phần ký ban hành, ngày 20/11/2019), không có Điều 220. Câu hỏi "bắt đầu được áp dụng" không khớp với cụm "hiệu lực thi hành" trong luật. |
| Selected `hybrid_k3_nb` | 0 | 0 | 9 | Mọi câu đều có đủ trích dẫn và điều kiện, ví dụ q24 có cả điều kiện "sau khi trích nộp BHXH, BHYT, BHTN, thuế TNCN". |

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:** 3 tool: `search` (retriever đã cấu hình, áp filter người dùng), `read_neighbours` và `final_answer` (status `answered`/`abstain`, answer, citations). Vòng lặp dừng khi model gọi `final_answer`. Khi đã đủ 6 lượt gọi tool thì bắt buộc gọi `final_answer`. Model trả lời bằng text thường thì tính là abstain (`no_tool_call_final`). Context được gom theo thứ tự agent nhìn thấy, bỏ chunk trùng, cắt theo budget 6000 ký tự. Citation không có trong context hoặc thiếu `[id]` trong answer thì câu trả lời bị đổi thành abstain (`invalid_citations`). Lỗi của tool được trả lại cho model và ghi vào trace (`tool_error`). Lỗi provider hoặc vector store ghi `status: "error"` và không thay bằng câu trả lời giả. Ở mọi lần chạy dưới đây: 0 lỗi.

Nhãn chấm tay ở `reviews.json`, có lý do từng câu trong trường `note`. Số liệu tổng hợp bằng [starter/agent_metrics.py](../../starter/agent_metrics.py).

| Cấu hình / split | Tool use | Số lượt gọi tool (phân bố) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Prompt cũ, hybrid nb=1 / dev | 18/18 | 1:15, 2:1, 3:1, 5:1 | 13/0/2 · **0.27** | 3/4 (0.75) | 2/2 | 3/3 | 13/15 |
| Prompt mới, hybrid nb=0 / dev | 18/18 | 1:14, 2:2, 3:1, 6:1 | 1/1/13 · **1.80** | 14/16 (0.88) | 15/15 | 3/3 | 0/15 |
| Prompt mới, hybrid nb=1 (**chọn**) / dev | 18/18 | 1:15, 2:1, 3:1, 6:1 | 0/1/14 · **1.93** | 15/16 (0.94) | 15/15 | 3/3 | 0/15 |
| Baseline: prompt mới, dense nb=0 / test | 12/12 | 1:10, 6:2 | 0/0/9 · **2.00** | 9/10 (0.90) | 10/10 | 2/3 | 0/9 |
| Selected: prompt mới, hybrid nb=1 / test | 12/12 | 1:9, 2:1, 5:2 | 1/0/8 · **1.78** | 8/9 (0.89) | 9/9 | 2/3 | 1/9 |

Chi tiết từng câu test cho cấu hình đã chọn. Baseline giống hệt, trừ q17 được 2 điểm.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q16 | trực tiếp | 1 search, `p040:000` rank 1 | "từ 22 giờ đến 06 giờ sáng ngày hôm sau [p040:000]" | 2 · khớp nguyên văn Điều 106 |
| q17 | diễn đạt lại | 2 search: "thời điểm áp dụng" (9 chunk không liên quan lấp đầy budget), rồi "hiệu lực thi hành" (tìm đúng `p083:001` nhưng không vào được context) | bị đổi thành abstain (`invalid_citations`); câu bị loại: "hiệu lực từ 01/01/2021 [p083:001]" | **0** · từ chối trên câu có đáp án. Xem giới hạn 1 |
| q18 | điều kiện | 1 search, `p041:003` | "bình quân 01 tháng ít nhất 04 ngày [p041:003]" | 2 · Điều 111 khoản 1 |
| q19 | metadata filter (chương 6) | 1 search, `p037:003` | "ít nhất 30% … ngày làm việc bình thường [p037:003]" | 2 · Điều 98 khoản 2 |
| q20 | trực tiếp | 1 search, `p060:003` | "62 tuổi vào năm 2028 [p060:003]" | 2 · Điều 169 khoản 2 |
| q21 | trực tiếp | 1 search, `p041:004` | "Tết Âm lịch 05 ngày, hưởng nguyên lương [p041:004]" | 2 · Điều 112 khoản 1 điểm b |
| q22 | diễn đạt lại | 1 search, `p042:003` | "đủ 05 năm … tăng thêm 01 ngày [p042:003]" | 2 · Điều 114 |
| q23 | điều kiện | 1 search, `p047:003` | "không quá 15 ngày, đặc biệt không quá 90 ngày [p047:003]" | 2 · Điều 128 |
| q24 | trực tiếp | 1 search, `p039:001` | "tối đa 30% … sau khi trích nộp BHXH, BHYT, BHTN, thuế TNCN [p039:001]" | 2 · Điều 102 khoản 3, đủ điều kiện |
| u04 | thiếu bằng chứng | 1 search | status `answered`: "hưởng chế độ thai sản theo pháp luật BHXH, nhưng không nêu cụ thể số ngày [p051:001]" | **sai** · nội dung là từ chối mềm, không bịa số ngày, nhưng status không phải abstain. Citation sai: `p051:001` chỉ có khoản 1–3 Điều 139 (lao động nữ) |
| u05 | thiếu bằng chứng | 5 search | abstain: không có mức hưởng trợ cấp thất nghiệp | đúng · không đưa con số nào |
| u06 | thiếu bằng chứng | 5 search | abstain: không có mức phạt hành chính | đúng · không đưa con số nào |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace, nhãn, số liệu: `results/agent/<split>/<tag>/` gồm `traces.json`, `cases.json`, `run.json`, `reviews.json`, `metrics.json`. Tag dev: `gpt4omini_hybrid_oldprompt`, `gpt4omini_hybrid_nb0`, `gpt4omini_hybrid`. Tag test: `baseline`, `selected`.
  - Thời gian và token (prompt/completion): dev chọn 64,8 s, 78.905/1.870 · test baseline 50,3 s, 50.134/1.335 · test selected 60,9 s, 75.737/1.324.
  - Chi phí **ước tính** theo giá niêm yết GPT-4o-mini ($0,15 / 1M input, $0,60 / 1M output): khoảng $0,013 cho một lượt dev, $0,008 cho test baseline, $0,012 cho test selected. Tổng các lần chạy trong báo cáo dưới $0,06. Script không tự tính chi phí (`cost_usd` = 0/None), con số thật xem trên dashboard OpenRouter.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:** trước/sau của hai lỗi dev xem ở phần Retrieval. **Trên test, cấu hình đã chọn thấp hơn baseline** (1.78 so với 2.00), dù ở tầng retrieval nó cao hơn (CER 9/9 so với 8/9). Báo cáo giữ nguyên kết quả này, không chỉnh cấu hình theo test.

## Kết luận và giới hạn

1. **Neighbours làm đầy budget và chặn evidence của lượt tìm sau.** Đây là nguyên nhân q17 bị 0 điểm trên test. `pack_context` giữ chunk theo thứ tự agent nhìn thấy. Lượt search đầu với query kém, cộng thêm 2 chunk lân cận cho mỗi hit, đã chiếm hết 6000 ký tự, nên chunk đúng của lượt 2 bị bỏ. Ở eval retrieval (dùng câu hỏi gốc) không có lỗi này, nên mới thấy ở tầng agent. Hướng sửa có thể là ưu tiên chunk của lượt tìm mới nhất, hoặc chỉ thêm neighbours cho top-1. **Chưa thử**, vì test đã dùng; muốn đánh giá độc lập cần bộ test mới.
2. **Citation chưa chính xác ở mức điều khoản** (q07 dev, q11 dev nb=0, u04 test): model trích chunk chứa một phần khẳng định, không phải chunk chứa điều kiện chính. Luật kiểm tra hiện tại chỉ xác nhận citation có trong context, không xác nhận nó hỗ trợ khẳng định. Phần đó phải chấm tay.
3. **Abstention dựa vào `status`.** u04 thực chất đã từ chối nhưng lại đặt `status="answered"`, nên bị tính sai ở cả hai cấu hình.
4. **Mẫu nhỏ:** dev 18 câu, test 12 câu. Mỗi câu chiếm khoảng 11% điểm test, nên chênh 2.00 so với 1.78 chỉ là **một câu**, không đủ để kết luận cấu hình nào tốt hơn.
5. **Một model, một lượt chạy** (temperature 0, không lặp lại). Prompt mới mới được kiểm chứng trên GPT-4o-mini. Model free `gemma-4-31b-it:free` bị rate limit (HTTP 429 từ pool chung) nên không có kết quả.
6. **Người chấm:** nhãn câu trả lời, citation và context được soạn nháp với trợ giúp của AI. Người học đã đối chiếu với gold và chunk rồi duyệt lại. Chỉ có một người chấm, không đo độ đồng thuận.

## Tái lập

```bash
pip install -r requirements.txt openai             # BM25 tự cài đặt trong src/day7/retrievers.py
cp .env.example .env    # điền OPENROUTER_API_KEY; LLM_PROVIDER=openrouter, LLM_MODEL=openai/gpt-4o-mini
cd starter && docker compose -p day6_student_starter up -d --wait && cd ..   # volume Day 6 đã có index
export PYTHONPATH=src PYTHONUTF8=1
python -m unittest discover -s tests                                   # test mock, không gọi mạng
python -m day7.eval_retrieval --split dev --configs all
python -m day7.run_agent --split dev  --retriever hybrid --neighbours 1 --tag gpt4omini_hybrid
python -m day7.eval_retrieval --split test --configs dense_k3,hybrid_k3_nb
python -m day7.run_agent --split test --retriever dense  --neighbours 0 --tag baseline
python -m day7.run_agent --split test --retriever hybrid --neighbours 1 --tag selected
D=results/agent/test/selected; python starter/agent_metrics.py --cases $D/cases.json --traces $D/traces.json --reviews $D/reviews.json --split test --output $D/metrics.json
```

- Chạy `docker compose` **phải có** `-p day6_student_starter` để gắn đúng volume đã có dữ liệu. Không có `-p` thì Docker tạo một DB trống.
- `starter/data/day6/` (PDF, OCR, chunk) không nằm trong Git. Có thể tái tạo từ repo Day 6 (https://github.com/kyo708/K5-L3A-DAY06-VuThanhToan-2A202603654-DataIngestion, commit `21e8f16`) bằng `python starter/import_day6.py --day6-chunks starter/data/day6/chunks.jsonl --manifest starter/data/day6/manifest.json --output starter/data/day6`.
- Lỗi dev 1 (prompt cũ) được chạy lại bằng một script tạm, thay quy tắc 4 lúc chạy (monkeypatch) mà không sửa file. Prompt cũ là: *"4. Khi trả lời, gọi `final_answer` với status="answered", câu trả lời ngắn gọn tiếng Việt, mỗi khẳng định kèm citation dạng [chunk_id] và liệt kê đúng các chunk_id đó trong `citations`."*

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
