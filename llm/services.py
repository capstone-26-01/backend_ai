import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_repo_code(repo_path, files, get_content_func):
    code_text = ""
    for file_path in files:
        if not file_path.endswith('.py'):
            continue
        code = get_content_func(repo_path, file_path)
        if code:
            code_text += f"\n\n# 파일: {file_path}\n{code}"
            
    #print(code_text)
    return code_text


def answer_question(repo_path, files, question, get_content_func):
    code_text = get_repo_code(repo_path, files, get_content_func)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "너는 코드 분석 전문가야. 주어진 코드를 분석해서 질문에 답해줘. 한국어로 답변해."
            },
            {
                "role": "user",
                "content": f"다음 코드를 분석해줘:\n{code_text}\n\n질문: {question}"
            }
        ]
    )

    return response.choices[0].message.content