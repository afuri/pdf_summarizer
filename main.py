import os
import sys
import json
import requests
from dotenv import load_dotenv
from pdfminer.high_level import extract_text


def extract_pdf_text(pdf_path):
    """Извлекает текст из PDF-файла с помощью pdfminer.six."""
    try:
        text = extract_text(pdf_path)
        if not text.strip():
            raise ValueError("PDF file contains no extractable text.")
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract PDF text: {e}")


def call_groq_api(api_key, text, prompt):
    """Отправляет запрос к Groq API и возвращает результат."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": f"You are a helpful assistant that extracts specific information from documents. The user wants you to: {prompt}\n\nPlease return your response as valid JSON only, with no additional text or formatting."
            },
            {
                "role": "user",
                "content": f"Here is the document text:\n\n{text}"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2000
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()
            try:
                content = response_json["choices"][0]["message"]["content"]
                return json.loads(content)
            except Exception as e:
                return {
                    "error": f"Failed to parse response: {e}",
                    "raw_response": response_json
                }
        else:
            return {
                "error": f"API call failed: {response.status_code}",
                "message": response.text
            }
    except Exception as e:
        return {"error": f"Request failed: {e}"}


def get_next_filename(base_name):
    if not os.path.exists(base_name):
        return base_name
    name, ext = os.path.splitext(base_name)
    i = 1
    while True:
        new_name = f"{name}_{i}{ext}"
        if not os.path.exists(new_name):
            return new_name
        i += 1


def main():
    load_dotenv()
    print("Программа PDF-Summarizer v.1.0")
    print("Программа использует LLM llama4 с ограничением 30000")
    pdf_file = input("Введите путь к  pdf файлу: ").strip()
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file '{pdf_file}' does not exist", file=sys.stderr)
        sys.exit(1)
    prompt = input("Введите запрос (promt): ").strip()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set in .env file", file=sys.stderr)
        sys.exit(1)
    try:
        text = extract_pdf_text(pdf_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    result = call_groq_api(api_key, text, prompt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    save = input("Вы хотите сохранить результат? Y/N\n").strip()
    if save.lower() == "y":
        filename = get_next_filename("result.json")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Результат сохранён в файл: {filename}")
        except Exception as e:
            print(f"Ошибка при сохранении файла: {e}", file=sys.stderr)
    else:
        print("Работа программы завершена.")


if __name__ == "__main__":
    main()
