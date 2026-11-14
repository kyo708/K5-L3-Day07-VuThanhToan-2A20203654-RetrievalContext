# Tái sử dụng từ Day 6

Sao chép nguyên văn từ repo Day 6 (`starter/src/`), commit `21e8f16`
(https://github.com/kyo708/K5-L3A-DAY06-VuThanhToan-2A202603654-DataIngestion).

| File | Nội dung |
|---|---|
| `embeddings.py` | `SentenceTransformerEmbedder`: `intfloat/multilingual-e5-base`, 768 chiều, normalize, prefix `query: ` / `passage: ` |
| `store.py` | `PgVectorStore`: bảng `chunks_768`, cosine `<=>`, exact search, filter `chapter`/`page`/`document_id` |
| `models.py` | dataclass `Chunk`, `DocumentPage` |

Vector store: PostgreSQL 16 + pgvector 0.8.7 (`starter/docker-compose.yml`, cổng `127.0.0.1:55433`).
Dùng cùng project name Day 6 để gắn lại volume đã có dữ liệu:

```powershell
cd starter
docker compose -p day6_student_starter up -d --wait
```

Index Day 6: chiến lược `A` (fixed 500/overlap 50, 452 chunk) và `B` (theo đoạn, max 1000, 343 chunk);
`C` = index `B` + filter `chapter`. Query embedding phải dùng `embed_query` của cùng model.

Dữ liệu (không theo dõi bằng Git, xem `.gitignore`): `starter/data/day6/`
- `blld.pdf`, `ocr/blld/` — corpus gốc + OCR (manifest: `data/manifest.json`)
- `chunks.jsonl` — chunk Day 6 gốc; `chunks_A.json`, `chunks_B.json` — schema Day 7 (`python starter/import_day6.py ...`)
- `questions_day6.json`, `metrics.json`, `run_config.json` — benchmark và kết quả Day 6 để tham chiếu
