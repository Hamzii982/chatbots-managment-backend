import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.graph.workflow import create_graph
from langchain_core.messages import AIMessageChunk
from app.db.session import SessionLocal  # your DB session factory

async def test_stream():
    print("=== Starting stream test ===\n")

    db = SessionLocal()  # real DB session

    try:
        async with AsyncSqliteSaver.from_conn_string("test_checkpoints.db") as checkpointer:
            graph = create_graph(checkpointer)

            config = {
                "configurable": {
                    "thread_id": "test-thread-1",
                    "db": db  # ← real session, same as API does it
                }
            }

            inputs = {
                "messages": [("user", "Hallo! Was kannst du?")],  # Simulated user message
                "chatbot_id": 1  # ← use a real chatbot ID from your DB
            }

            print("Waiting for chunks...\n")
            chunk_count = 0
            last_time = asyncio.get_event_loop().time()

            # Use astream_events with version="v2"
            async for event in graph.astream_events(inputs, config, version="v2"):
                kind = event["event"]

                # This event triggers for every single new token/word
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        print(content, end="", flush=True)

                # Optional: See when nodes start/end
                elif kind == "on_chain_start" and event["name"] == "model":
                    print("\n[Model started thinking...]\n")
                elif kind == "on_chain_end" and event["name"] == "model":
                    print("\n\n[Model finished]")

            print(f"\n=== Done. Got {chunk_count} chunks from 'model' node ===")

            if chunk_count <= 1:
                print("❌ NOT streaming — everything came in one chunk")
                print("   → chain.astream() in call_model is still buffering")
            else:
                print("✅ Streaming works at graph level")
                print("   → Problem is in FastAPI/middleware layer")

    finally:
        db.close()

asyncio.run(test_stream())

async def test_llm_direct():
    print("=== Testing LLM directly (no LangGraph) ===\n")
    
    from app.models.chatbot import Chatbot
    from app.models.model_config import ModelConfig
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate

    db = SessionLocal()
    chatbot_id = 1  # your real chatbot ID

    chatbot = db.query(Chatbot).get(chatbot_id)
    model_cfg = db.query(ModelConfig).get(chatbot.model_id)

    print(f"Provider: {model_cfg.provider}")
    print(f"Model: {model_cfg.model_name}\n")

    provider = model_cfg.provider.lower()
    if provider == "openai":
        llm = ChatOpenAI(
            model=model_cfg.model_name,
            temperature=model_cfg.temperature,
            api_key=model_cfg.api_key,
            streaming=True,
        )
    elif provider == "anthropic":
        llm = ChatAnthropic(
            model=model_cfg.model_name,
            temperature=model_cfg.temperature,
            api_key=model_cfg.api_key,
            streaming=True,
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", chatbot.system_prompt),
        ("placeholder", "{messages}"),
    ])

    chain = prompt | llm

    chunk_count = 0
    import time
    last = time.time()

    async for chunk in chain.astream({
        "messages": [("user", "Hallo! Was kannst du?")],
    }):
        now = time.time()
        if chunk.content:
            print(f"[+{round(now - last, 3)}s] {repr(chunk.content)}")
            chunk_count += 1
            last = now

    print(f"\n=== Got {chunk_count} chunks ===")
    if chunk_count <= 1:
        print("❌ LLM itself is not streaming — check api_key, model name, or langchain version")
    else:
        print("✅ LLM streams fine — problem is inside LangGraph node setup")

    db.close()

# asyncio.run(test_llm_direct())

async def test_langgraph_stream():
    db = SessionLocal()

    async with AsyncSqliteSaver.from_conn_string("test_checkpoints.db") as checkpointer:
        graph = create_graph(checkpointer)
        config = {"configurable": {"thread_id": "debug-1", "db": db}}
        inputs = {"messages": [("user", "Hello!")], "chatbot_id": 1}

        # Test updates mode
        print("--- stream_mode=updates ---")
        async for update in graph.astream(inputs, config, stream_mode="updates"):
            for node_name, node_output in update.items():
                print(f"  Node={node_name} output_keys={list(node_output.keys()) if isinstance(node_output, dict) else type(node_output)}")

        # Test messages mode
        print("\n--- stream_mode=messages ---")
        count = 0
        async for msg, metadata in graph.astream(inputs, config, stream_mode="messages"):
            count += 1
            print(f"  chunk={count} node={metadata.get('langgraph_node')} type={type(msg).__name__} content={repr(getattr(msg, 'content', ''))[:60]}")

    db.close()

# asyncio.run(test_langgraph_stream())

async def test_stream_no_checkpointer():
    print("=== Stream test WITHOUT checkpointer ===\n")

    db = SessionLocal()

    try:
        # ✅ No checkpointer — MemorySaver or nothing
        graph = create_graph(checkpointer=None)

        config = {
            "configurable": {
                "thread_id": "test-thread-1",
                "db": db
            }
        }

        inputs = {
            "messages": [("user", "Hallo! Was kannst du?")],
            "chatbot_id": 1
        }

        chunk_count = 0
        print("Waiting for chunks...\n")

        async for msg, metadata in graph.astream(inputs, config, stream_mode="messages"):
            node = metadata.get("langgraph_node")
            if node == "model" and isinstance(msg, AIMessageChunk) and msg.content:
                chunk_count += 1
                print(f"[chunk #{chunk_count}] {repr(msg.content)}")

        print(f"\n=== Done. Got {chunk_count} chunks ===")
        if chunk_count > 1:
            print("✅ Streaming works — AsyncSqliteSaver was the problem")
        else:
            print("❌ Still buffering — problem is NOT the checkpointer")

    finally:
        db.close()

# asyncio.run(test_stream_no_checkpointer())

async def test_raw_chain():
    
    from app.models.chatbot import Chatbot
    from app.models.model_config import ModelConfig
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    
    print("=== Raw chain stream (zero LangGraph) ===\n")

    db = SessionLocal()
    try:
        chatbot = db.query(Chatbot).get(1)
        model_cfg = db.query(ModelConfig).get(chatbot.model_id)

        print(f"Provider: {model_cfg.provider}")
        print(f"Model: {model_cfg.model_name}\n")

        llm = ChatOpenAI(
            model=model_cfg.model_name,
            temperature=model_cfg.temperature,
            api_key=model_cfg.api_key,
            streaming=True,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", chatbot.system_prompt),
            ("human", "{input}"),
        ])

        chain = prompt | llm

        chunk_count = 0
        async for chunk in chain.astream({"input": "Hallo! Was kannst du?"}):
            if chunk.content:
                chunk_count += 1
                print(f"[chunk #{chunk_count}] {repr(chunk.content)}")

        print(f"\nTotal: {chunk_count} chunks")
        if chunk_count > 1:
            print("✅ Chain streams fine — problem is inside LangGraph node execution")
        else:
            print("❌ Chain itself is buffering — LangChain/OpenAI issue")

    finally:
        db.close()

# asyncio.run(test_raw_chain())