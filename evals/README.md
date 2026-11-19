# Bộ đánh giá prototype 300 câu

Các file đã đóng băng trong repo, chạy evaluation không cần đọc lại `DSC_Legal/QA`.

| Bộ | Số câu | Nội dung |
|---|---:|---|
| `datasets/qa250_v1/cases.jsonl` | 250 | 200 câu từ QA/train, 50 câu từ QA/warmup; giữ đáp án riêng của từng record |
| `datasets/guardrails50_v1/cases.jsonl` | 50 | Case hành vi do dự án biên soạn, có rubric riêng |
| `datasets/prototype300_v1/cases.jsonl` | 300 | Ghép nguyên hai bộ trên |

## Chọn câu QA

Seed 2026; chỉ nhận câu hỏi và đáp án là chuỗi có nội dung, không chuyển
`answer: null` thành đáp án. Không dùng public chưa có nhãn. Chuẩn hóa NFC,
chữ hoa/thường và khoảng trắng để loại câu hỏi trùng. Loại cả các câu trong
hai bộ đánh giá 30/50 cũ. Khi câu xuất hiện ở nhiều nguồn, nguồn đầu tiên được
ưu tiên; phần warmup được lấy từ những câu còn lại sau khử trùng toàn bộ train.

Lấy luân phiên theo nhóm loại câu và chủ đề để tăng độ đa dạng. Nhóm được gán
bằng heuristic có thể kiểm tra được, không phải nhãn độ khó đã được chuyên gia
xác nhận:

| Nhóm QA | Số câu |
|---|---:|
| Thông thường | 93 |
| Đáp án tham chiếu có ít nhất hai số Điều khác nhau | 93 |
| Câu hỏi có năm/mốc thời gian/từ chỉ hiệu lực | 64 |

Chủ đề gồm lao động, đất đai/nhà ở, dân sự/gia đình, kinh doanh/thuế, hình sự/tố
tụng, giao thông, giáo dục/y tế và hành chính/khác. Chủ đề cũng chỉ là heuristic.
`source_file`, `source_query_id` xác định record gốc; manifest ghi hash từng
nguồn, danh sách ID, các trường hợp bị loại và SHA-256 của `cases.jsonl`.

Đáp án gốc được giữ nguyên nội dung sau strip khoảng trắng ở hai đầu và đánh dấu
`reference_quality=unreviewed`. Chúng phản ánh dataset, chưa được xác minh lại
về tính đúng hoặc hiệu lực pháp luật hiện tại. Bộ này phục vụ development; cần
một tập holdout mới trước khi công bố kết quả tổng quát.

## 50 case guardrails/abstention

| Nhóm | Số câu | Hành vi mong đợi |
|---|---:|---|
| Prompt injection, lộ bí mật, bịa nguồn | 10 | `safe_response` |
| Thiếu dữ kiện thiết yếu | 10 | `clarify` |
| Thông tin/hồ sơ không được cung cấp hoặc không thể truy cập | 10 | `abstain` |
| Ngoài phạm vi trợ lý pháp luật | 10 | `abstain` |
| Hỏi tiếp nhưng không có lịch sử | 10 | `clarify` |

Mỗi câu có `reference_answer` là rubric hành vi, không phải đáp án pháp lý giả
lập. `accepted` ở nhóm này xác nhận rubric do dự án biên soạn, không phải đã
được chuyên gia pháp lý duyệt. Không suy ra rằng hệ thống hiện đã pass các case
này. API hiện stateless; nhóm hỏi tiếp kiểm tra khả năng nhận ra thiếu ngữ cảnh,
không đánh giá hội thoại nhiều lượt thật. Nhóm ngoài phạm vi thể hiện policy
sản phẩm mong muốn, không phải policy đã được code enforce. Chỉ dẫn tấn công ở
đây nằm trong câu hỏi; chưa kiểm tra injection trong payload tài liệu truy hồi.

Judge v2 nhận `expected_behavior` và rubric, không đòi citation cho một lời từ
chối/hỏi lại đúng. Report tách `slice:*`, số câu đã được chấm và độ bao phủ;
điểm judge không thay thế human review về chất lượng pháp lý.

## Tái tạo

Chạy vào thư mục output mới để giữ các bộ hiện có:

```bash
legal-rag-eval prepare \
  --source ../DSC_Legal/QA/train.json \
  --source ../DSC_Legal/QA/warmup.json \
  --source-counts 200 50 --sample-size 250 --seed 2026 --strategy stratified \
  --exclude-dataset evals/datasets/qa_random30_v0/cases.jsonl \
  --exclude-dataset evals/datasets/qa_random50_v0_2082005/cases.jsonl \
  --output /tmp/qa250_recreated

legal-rag-eval prepare-guardrails --output /tmp/guardrails50_recreated

legal-rag-eval combine \
  --dataset /tmp/qa250_recreated/cases.jsonl \
  --dataset /tmp/guardrails50_recreated/cases.jsonl \
  --output /tmp/prototype300_recreated
```

Các `cases.jsonl` được tái tạo phải có cùng hash khi dùng cùng source/version
code/seed; timestamp, tên bộ và đường dẫn trong manifest có thể khác.
