# 랭그래프로 에이전트 그래프 생성하기
# 01. 타빌리 서치 도구 및 LLM 설정하기
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import json
from langchain.messages import ToolMessage


load_dotenv()

tool = TavilySearch(max_results=3)
tools = [tool]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# 02. 상태그래프 생성하기
# 메세지 목록을 관리하는 그래프 상태 정의하고 상태그래프 만들기
class State(TypedDict):
    messages: Annotated[list, add_messages]
    
graph_builder = StateGraph(State)

# 03. 노드 생성하기
# LLM의 답변을 생성하는 노드 만들기
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)

# 04. 노드 생성하기 (도구 노드)
# LLM이 호출한 도구를 실행하는 노드 만들기
class BasicToolNode:
    """
        마지막 AIMessage에서 요청된 도구를 실행하는 노드
    """
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
        
    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("ERROR: 입력에 메시지가 없습니다.")
        
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# 05. 엣지 생성하기 (조건부 흐름 구성)
# LLM의 도구 호출 결과에 따라 처리하는 조건부 엣지 만들기
def route_tools(state: State):
    """
    마지막 메시지에 도구 호출이 있는 경우, ToolNode로 라우팅하고 그렇지 않으면 END로 라우팅
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"ERROR: 입력에 메시지가 없습니다. 상태: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END}, 
)

# 06. 엣지 생성 및 그래프 컴파일하기
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

# pretty_print()를 사용해 메시지 목록 출력하기
def invoke():
    response = graph.invoke(
        {
            "messages": ["Langgraph가 무엇인가요?"]
        }
    )
    
    for msg  in response["messages"]:
        msg.pretty_print()

# ainvoke()를 사용해 실행 결과 확인하기
async def ainvoke():
    response = await graph.ainvoke(
        {
            "messages": ["Langgraph가 무엇인가요?"]
        }
    )
    
    for msg in response["messages"]:
        msg.pretty_print()
        
# stream()의 updates 모드를 활용해 실행 결과 확인하기
def stream():
    response = graph.stream(
        {
            "messages": ["Langgraph가 무엇인가요?"]
        }
    )
    for chunk in response:
        for node, state in chunk.items():
            print("---", node, "---")
            print(state)
            print("=" * 60)

# stream()의 values 모드를 활용해 실행 결과 확인하기 - 사실 상 이 모드가 제일 많이 쓰일듯
def stream_values():
    response = graph.stream(
        {
            "messages": ["Langgraph가 무엇인가요?"]
        },
        stream_mode="values"
    )
    
    for chunk in response:
        for state_key, state_value in chunk.items():
            print("=== 현재 상태 ===")
            for msg in state_value:
                print(f"{type(msg).__name__}: {msg.content[:50]}")
            if state_key == "messages":
                state_value[-1].pretty_print()
            print("=" * 60)

# stream()의 messages 모드를 활용해 실행 결과 확인하기
def stream_messages():
    response = graph.stream(
        {
            "messages": ["Langgraph가 무엇인가요?"]
        },
        stream_mode="values"
    )
    
    for token, metadata in response:
        print(token)
        # print(metadata["langgraph_node"])
        
# astream()를 활용해 실행 결과 확인하기
async def astream():
    response = graph.astream(
        {
            "messages": ["Langgraph가 무엇인가요?"]
        }
    )
    async for chunk in response:
        for node, state in chunk.items():
            print("---", node, "---")
            print(state)
            print("=" * 60)


if __name__ == "__main__":
    try:
        image = graph.get_graph().draw_mermaid_png()
        with open("web_agent/graph.png", "wb") as f:
            f.write(image)
    except Exception:
        pass
    
    # invoke()
    
    # import asyncio
    # asyncio.run(ainvoke())
    
    # stream()
    
    # stream_values()
    
    # stream_messages()
    
    import asyncio
    asyncio.run(astream())