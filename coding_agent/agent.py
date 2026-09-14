from dotenv import load_dotenv

# `create_agent`는 두 가지 필수 파라미터만 전달하면 도구 호출을 기반으로 동작하는 에이전트를 생성할 수 있음.
# model: 에이전트가 사용할 LLM
# tools: 에이전트가 사용할 도구의 목록
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools import python_exec_tool


load_dotenv()

tools = [python_exec_tool]

llm = ChatOpenAI(model='gpt-4o')
graph = create_agent(llm, tools)


# 에이전트 그래프를 시각화하여 PNG 파일로 저장하는 함수
def save_graph_png(path: str = "coding_agent/graph.png") -> None:
    """
    에이전트의 그래프 구조를 Mermaid 이미지로 렌더링하여 PNG 파일로 저장합니다.

    Args:
        path: 저장할 PNG 파일 경로
    """
    image = graph.get_graph().draw_mermaid_png()
    with open(path, "wb") as f:
        f.write(image)


if __name__ == "__main__":
    save_graph_png()

    response = graph.stream(
        {
            "messages": [
                "첫 번째 항이 1인 피보나치 후열을 출력하는 파이썬 코드를 작성해주세요."
            ]
        }
    )
    
    for chunk in response:
        for node, value in chunk.items():
            if node:
                print("---", node, "---")
            if "messages" in value:
                print(value["messages"][0].content)