import asyncio
import os

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
# 移除不正确的导入
# from mcp_agent.mcp.registry import get_server_registry

app = MCPApp(name="hello_world_agent")

async def example_usage():
    async with app.run() as mcp_agent_app:
        logger = mcp_agent_app.logger
        
        # 先不检查可用服务器，直接尝试创建agent
        finder_agent = Agent(
            name="finder",
            instruction="""You can read local files.
                Return the requested information when asked.""",
            server_names=["filesystem"],  # 只使用文件系统服务器
        )

        async with finder_agent:
            # 自动初始化MCP服务器并为LLM添加工具
            tools = await finder_agent.list_tools()
            logger.info(f"Tools available:", data=tools)

            # 附加OpenAI LLM到agent
            llm = await finder_agent.attach_llm(OpenAIAugmentedLLM)

            # 读取README.md文件
            result = await llm.generate_str(
                message="Show me what's in README.md verbatim"
            )
            logger.info(f"README.md contents: {result}")

            # 由于没有fetch服务器，改为询问本地文件内容相关问题
            result = await llm.generate_str(
                message="Analyze the README.md content and tell me what this project does"
            )
            logger.info(f"Analysis: {result}")

            # 多轮对话示例
            result = await llm.generate_str("Summarize that in a 128-char tweet")
            logger.info(f"Tweet: {result}")

if __name__ == "__main__":
    asyncio.run(example_usage())