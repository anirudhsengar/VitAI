from fastmcp import FastMCP
from agent import create_agent

mcp = FastMCP("VitAI MCP Server")

# Default Adoptium repositories that the agent is allowed to query
DEFAULT_REPOSITORIES = [
    "adoptium/aqa-tests",           # Central project for AQAvit
    "adoptium/TKG",                 # Lightweight test harness for running diverse tests
    "adoptium/aqa-systemtest",      # System verification tests
    "adoptium/aqa-test-tools",      # Various test tools that improve workflow
    "adoptium/STF",                 # System Test Framework
    "adoptium/bumblebench",         # Microbenchmarking test framework
    "adoptium/run-aqa",             # run-aqa GitHub action
    "adoptium/openj9-systemtest",   # System verification tests for OpenJ9
    "eclipse-openj9/openj9",        # OpenJ9 JVM
]

_agent = None

def get_agent():
    """Get or create agent instance with specified repositories."""
    global _agent
    if _agent is None:
        _agent = create_agent(repositories=DEFAULT_REPOSITORIES, max_iterations=10)
    return _agent

@mcp.tool
def query(input: str) -> str:
    """
    Ask VitAI questions related to Adoptium repositories and it will begin exploring 
    GitHub and provide grounded answers based on the code present in the repositories.
    
    The agent has access to the following Adoptium/OpenJ9 repositories:
    - adoptium/aqa-tests: Central project for AQAvit
    - adoptium/TKG: Lightweight test harness
    - adoptium/aqa-systemtest: System verification tests
    - adoptium/aqa-test-tools: Test workflow tools
    - adoptium/STF: System Test Framework
    - adoptium/bumblebench: Microbenchmarking framework
    - adoptium/run-aqa: GitHub action for running AQA
    - adoptium/openj9-systemtest: OpenJ9 system tests
    - eclipse-openj9/openj9: OpenJ9 JVM implementation
    
    Args:
        input: Your question about the repositories
    
    Returns:
        A detailed answer based on code and issues found in the repositories
    """

    repositories = DEFAULT_REPOSITORIES
    
    agent = get_agent()
    
    try:
        answer = agent.query(input, repositories)
        return answer
    except Exception as e:
        return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    mcp.run()