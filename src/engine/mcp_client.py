import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def async_search_mcp(query_text: str) -> str:
    """Asynchronously connects to the MCP server and calls the search tool."""
    
    # 1. Define exactly where the server lives
    server_params = StdioServerParameters(
        command="D:\\llm_arena\\llmev\\python.exe",
        args=["D:\\llm_arena\\src\\engine\\mcp_server.py"]
    )
    
    try:
        # 2. Open the connection
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                
                # 3. Initialize the protocol
                await session.initialize()
                
                # 4. Call the exact tool we built earlier
                result = await session.call_tool(
                    "search_knowledge_base", 
                    arguments={"query": query_text, "top_k": 3}
                )
                
                # Extract the text from the server's response
                if result.content and len(result.content) > 0:
                     return result.content[0].text
                return "No content returned from server."
                
    except Exception as e:
        return f"MCP Connection Error: {str(e)}"

def get_context_from_mcp(query: str) -> str:
    """
    Synchronous wrapper for Streamlit. 
    Streamlit can just call this normal function!
    """
    return asyncio.run(async_search_mcp(query))

# Quick test if you run this file directly
if __name__ == "__main__":
    print("Testing connection to MCP Server...")
    response = get_context_from_mcp("testing")
    print("\n--- SERVER RESPONSE ---")
    print(response)