# Phân tích điểm yếu của dự án RAG Chatbot cơ bản và Hướng cải thiện

Dự án RAG Chatbot trong tài liệu là một phiên bản "Native" (thuần túy) rất tốt để học hỏi và hiểu luồng hoạt động cơ bản. Tuy nhiên, nếu mang vào môi trường thực tế (Production), hệ thống này bộc lộ rất nhiều điểm yếu. 

Dưới đây là phân tích chi tiết các điểm yếu và hướng cải thiện tương ứng cho từng thành phần trong Pipeline:

---

## 1. Khâu Đọc và Trích xuất tài liệu (Data Ingestion)

### Điểm yếu:
- **Hạn chế của thư viện `pypdf`:** Thư viện này chỉ trích xuất được văn bản thuần (plain text). Nó **hoàn toàn mù tịt** trước:
  - Bảng biểu (Tables): Cấu trúc hàng/cột sẽ bị phá vỡ thành một mớ ký tự lộn xộn.
  - Hình ảnh, biểu đồ (Charts/Images): Sẽ bị bỏ qua hoàn toàn.
  - Layout phức tạp: Các văn bản chia làm 2 cột (như bài báo khoa học) thường bị đọc chéo từ cột này sang cột kia một cách sai lệch.

### Hướng cải thiện:
- Sử dụng các thư viện mạnh mẽ hơn như `pdfplumber` (giữ được cấu trúc bảng), `PyMuPDF` (nhanh và chính xác hơn).
- Với tài liệu chứa hình ảnh và bảng biểu phức tạp, sử dụng các công cụ chuyên dụng như **LlamaParse**, **Unstructured.io** hoặc kết hợp **OCR (Tesseract, GPT-4o, Claude 3.5 Sonnet)** để chuyển hình ảnh/bảng thành markdown trước khi xử lý.

---

## 2. Khâu Băm nhỏ văn bản (Chunking)

### Điểm yếu:
- **Cắt theo ký tự một cách mù quáng (Fixed-size Chunking):** Hàm `chunk_text()` trong bài cắt bằng cách cộng độ dài các chuỗi và dùng `overlap=200` ký tự. Cách này không quan tâm đến cấu trúc ngữ nghĩa:
  - Nó có thể vô tình cắt đứt đôi một câu đang dang dở.
  - Nó có thể chia cắt phần "Tiêu đề" (Heading) nằm ở chunk 1, nhưng "Nội dung" của tiêu đề đó lại nằm ở chunk 2, làm mất ngữ cảnh.

### Hướng cải thiện:
- **Semantic Chunking / Recursive Character Chunking:** Sử dụng các phương pháp cắt thông minh (như của thư viện LangChain) ưu tiên cắt theo dấu chấm câu (`.`), xuống dòng (`\n\n`), thay vì chỉ đếm số lượng ký tự.
- **Markdown/Header-aware Chunking:** Nhận diện các thẻ Header (H1, H2, H3) để gom nhóm các văn bản thuộc cùng một phần (Section) lại với nhau.

---

## 3. Khâu Vector hóa và Lưu trữ (Embedding & Vector DB)

### Điểm yếu:
- **Mất dữ liệu mỗi lần chạy lại (In-Memory Database):** Trong file `chatbot_app_native.py`, `chromadb.Client()` lưu dữ liệu trên RAM. Nghĩa là tắt web đi mở lại, toàn bộ Vector bị xóa.
- **Xử lý thừa thãi:** Nếu bạn upload cùng 1 file PDF 2 lần, hệ thống sẽ sinh ra 2 Collection khác nhau (vì đặt tên theo timestamp `rag_{time}`), gây tốn RAM và thời gian chạy.
- **Độ chính xác của Embedding tiếng Việt:** `bge-m3` rất tốt, nhưng với các từ khóa quá đặc thù của doanh nghiệp/ngành hẹp, Vector sinh ra có thể không sát nghĩa.

### Hướng cải thiện:
- **Persistent Storage:** Đổi sang `chromadb.PersistentClient(path="./chroma_db")` để lưu database xuống ổ cứng.
- **Quản lý File thông minh:** Băm (Hash) nội dung file PDF. Nếu file đã từng được upload, chỉ cần load lại database đã lưu thay vì ngồi cắt và embed lại từ đầu.
- Có thể dùng các model embedding đa ngôn ngữ mạnh hơn hoặc trả phí (OpenAI text-embedding-3) để có vector xịn hơn.

---

## 4. Khâu Tìm kiếm (Retrieval)

### Điểm yếu:
- **Tìm kiếm vector thuần túy đôi khi kém hiệu quả:** Tìm kiếm theo khoảng cách Vector (Cosine similarity) giỏi tìm "ngữ nghĩa" (semantic) nhưng lại rất "ngu" trong việc tìm **chính xác từ khóa** (keyword/exact match) như tên riêng, mã số tài liệu, ID khách hàng.
- **Bối cảnh bị hẹp (k=4):** Chỉ lấy 4 chunk ngắn nhất có thể không đủ để LLM trả lời các câu hỏi mang tính tổng hợp (cần đọc cả chương sách).

### Hướng cải thiện:
- **Hybrid Search (Tìm kiếm lai):** Kết hợp Vector Search (tìm theo ý nghĩa) + BM25 (tìm theo từ khóa chính xác). ChromaDB hỗ trợ làm việc này.
- **Reranking:** Sau khi lấy ra top 10 chunks, dùng thêm một AI thứ hai (gọi là Cross-Encoder / Reranker) để chấm điểm lại sự liên quan một cách tỉ mỉ nhất, rồi mới lấy 4 chunks đưa cho LLM. Phương pháp này tăng độ chính xác lên cực kỳ cao.

---

## 5. Sinh câu trả lời & Lịch sử hội thoại (LLM & Chat Memory)

### Điểm yếu:
- **Mô hìnhVicuna-7B:** Tiếng Việt chỉ ở mức cơ bản, có thể hành văn lủng củng và chạy khá chậm nếu máy tính không có Card màn hình (GPU) rời.
- **Chatbot "Mất trí nhớ" (Lỗi logic nghiêm trọng trong bài code Streamlit):** Mặc dù giao diện có hiển thị lịch sử chat (nhờ `st.session_state.chat_history`), nhưng khi gọi `ollama.chat()`, code hiện tại **chỉ truyền mỗi câu hỏi hiện tại (`q`)** vào Prompt. LLM hoàn toàn mù tịt về việc người dùng vừa hỏi gì câu trước đó. Bạn không thể hỏi bồi thêm kiểu *"Thế còn phần tiếp theo của nó thì sao?"*.
- **Không trích dẫn nguồn:** LLM trả lời xong người dùng không biết thông tin đó được lấy từ Trang số mấy của PDF để đối chứng.

### Hướng cải thiện:
- **Đổi LLM:** Dùng các mô hình mở xử lý tiếng Việt tốt và tối ưu hơn (như `Qwen2.5:7b`, `Gemma-2-9b` hoặc `Llama-3.1-8b`).
- **Xây dựng Conversational Memory:** Gộp các câu chat cũ vào chung với mảng `messages` khi gọi API của Ollama, hoặc dùng kỹ thuật *Condense Question* (dùng LLM viết lại câu hỏi mới dựa trên lịch sử trước khi đi tìm kiếm Vector).
- **Hiển thị Source (Nguồn):** Ở khâu `collection.add()`, lưu thêm metadata (chứa thông tin Trang số mấy). Khi LLM trả lời, in ra giao diện Streamlit: *"Nguồn tham khảo: Trang 15, Trang 22"*.
