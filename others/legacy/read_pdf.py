import fitz

doc = fitz.open(r'h:\Thành\Thành AI\AIVN\2026\Module\Module 1\project\2026\Documents_2026-5_Giao trước 2 project cuối module 1_Project-1.2.pdf')
text = '\n'.join([page.get_text() for page in doc])

with open(r'h:\Thành\Thành AI\AIVN\2026\Module\Module 1\project\2026\pdf_text.txt', 'w', encoding='utf-8') as f:
    f.write(text)
