import os
import json
from typing import List, Dict, Any, Optional, Tuple
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from search import GitHubSearchClient
import gradio as gr

load_dotenv()


class ReActAgentWithUI:
    """
    ReAct (Reasoning + Acting) agent that uses GitHub search tools with UI tracking.
    The agent follows the ReAct pattern: Thought -> Action -> Observation -> Repeat
    """
    
    def __init__(self, repositories: Optional[List[str]] = None, max_iterations: int = 5):
        """
        Initialize the ReAct agent.
        
        Args:
            repositories: List of repositories to search (format: 'owner/repo')
            max_iterations: Maximum number of reasoning iterations
        """
        self.token = os.getenv("GITHUB_TOKEN")
        self.endpoint = "https://models.github.ai/inference"
        self.model = "microsoft/Phi-4"
        self.max_iterations = max_iterations
        self.repositories = repositories or []
        
        # Initialize the LLM client
        self.llm_client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.token),
        )
        
        # Initialize the GitHub search client
        self.search_client = GitHubSearchClient(token=self.token)
        
        # System prompt for ReAct pattern
        self.system_prompt = """You are a helpful AI assistant that uses the ReAct (Reasoning + Acting) pattern to answer questions about Adoptium GitHub repositories.

You have access to the following tools:
1. search_code: Search for code in GitHub repositories
   - Parameters: query (str), repos (list of repo names in format owner/repo)
   - Returns: Code search results with file paths, repository names, and code snippets
   
2. search_issues: Search for issues in GitHub repositories
   - Parameters: query (str), repos (list of repo names in format owner/repo)
   - Returns: Issue search results with titles, descriptions, and metadata

Follow the GitHub search syntax rules:
- Use qualifiers like 'language:', 'extension:', 'path:', 'filename:' for code search
- Use qualifiers like 'is:issue', 'is:pr', 'state:open', 'label:' for issue search
- Combine terms with spaces for AND logic, use OR for alternatives

Use the ReAct pattern to solve problems:
1. Thought: Reason about what information you need and which tool to use
2. Action: Specify the tool and parameters in JSON format: {"tool": "tool_name", "parameters": {...}}
3. Observation: Analyze the results from the tool
4. Repeat until you have enough information to answer

When you have sufficient information, provide a final answer starting with "Final Answer:".

Important:
- Each action must be valid JSON on a single line
- Be specific in your search queries
- If results are insufficient, refine your search"""

    def _call_llm(self, messages: List[Any]) -> str:
        """Call the LLM with the given messages."""
        response = self.llm_client.complete(
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            max_tokens=1500,
            model=self.model
        )
        return response.choices[0].message.content

    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        Execute the specified tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute ('search_code' or 'search_issues')
            parameters: Parameters for the tool
            
        Returns:
            JSON string containing the results
        """
        try:
            repos = parameters.get('repos', self.repositories)
            query = parameters.get('query', '')
            
            # Build qualifiers for repository filtering
            qualifiers = {}
            if repos:
                # GitHub search supports multiple repo qualifiers
                qualifiers['repo'] = repos
            
            if tool_name == 'search_code':
                results = self.search_client.search_code(
                    q=query,
                    qualifiers=qualifiers,
                    per_page=10,
                    page=1
                )
                
                # Extract relevant information
                items = results.get('items', [])[:10]
                simplified_results = []
                for item in items:
                    simplified_results.append({
                        'name': item.get('name'),
                        'path': item.get('path'),
                        'repository': item.get('repository', {}).get('full_name'),
                        'html_url': item.get('html_url'),
                        'score': item.get('score')
                    })
                
                return json.dumps({
                    'total_count': results.get('total_count', 0),
                    'items': simplified_results
                }, indent=2)
                
            elif tool_name == 'search_issues':
                results = self.search_client.search_issues(
                    q=query,
                    qualifiers=qualifiers,
                    per_page=10,
                    page=1
                )
                
                # Extract relevant information
                items = results.get('items', [])[:10]
                simplified_results = []
                for item in items:
                    simplified_results.append({
                        'title': item.get('title'),
                        'number': item.get('number'),
                        'state': item.get('state'),
                        'repository': item.get('repository_url', '').split('/')[-2:],
                        'html_url': item.get('html_url'),
                        'body': item.get('body', '')[:200] + '...' if item.get('body') else None,
                        'labels': [label.get('name') for label in item.get('labels', [])]
                    })
                
                return json.dumps({
                    'total_count': results.get('total_count', 0),
                    'items': simplified_results
                }, indent=2)
            
            else:
                return json.dumps({'error': f'Unknown tool: {tool_name}'})
                
        except Exception as e:
            return json.dumps({'error': str(e)})

    def _parse_action(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse action from the agent's response.
        Looks for JSON in the format: {"tool": "tool_name", "parameters": {...}}
        
        IMPORTANT: Does not parse JSON that appears after "Final Answer:"
        """
        # Don't parse actions if we have a final answer
        if "Final Answer:" in text or "final answer:" in text.lower():
            return None
        
        try:
            # Try to find JSON in the text
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx + 1]
                action = json.loads(json_str)
                
                if 'tool' in action and 'parameters' in action:
                    return action
            
            return None
        except json.JSONDecodeError:
            return None

    def query_with_steps(self, user_query: str, repositories: Optional[List[str]] = None):
        """
        Process a user query using the ReAct pattern and yield step-by-step results.
        
        Args:
            user_query: The user's question
            repositories: Optional list of repositories to override the default
            
        Yields:
            Tuples of (step_output, final_output) for each iteration
        """
        # Use provided repositories or fall back to instance repositories
        if repositories:
            self.repositories = repositories
        
        # Initialize conversation with system prompt and user query
        messages = [
            SystemMessage(content=self.system_prompt),
            UserMessage(content=f"Question: {user_query}\n\nAvailable repositories: {', '.join(self.repositories)}")
        ]
        
        iteration = 0
        steps_output = []
        final_output = ""
        
        # Initial status
        yield (
            f"🤔 **Starting ReAct Process**\n\n**Query:** {user_query}\n\n**Available Repositories:**\n" + 
            "\n".join([f"- {repo}" for repo in self.repositories]) + "\n\n---\n",
            ""
        )
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Get response from LLM
            response = self._call_llm(messages)
            
            # Add iteration header
            iteration_output = f"\n### 🔄 Iteration {iteration}\n\n"
            iteration_output += f"**Agent Response:**\n```\n{response}\n```\n\n"
            
            steps_output.append(iteration_output)
            
            # Check if we have a final answer FIRST (before trying to parse actions)
            if "Final Answer:" in response or "final answer:" in response.lower():
                # Extract the final answer
                if "Final Answer:" in response:
                    final_answer = response.split("Final Answer:")[-1].strip()
                else:
                    # Case-insensitive search
                    idx = response.lower().find("final answer:")
                    final_answer = response[idx + len("final answer:"):].strip()
                
                final_output = f"## ✅ Final Answer\n\n{final_answer}"
                
                yield (
                    "".join(steps_output),
                    final_output
                )
                return
            
            # Try to parse an action (only if we don't have a final answer)
            action = self._parse_action(response)
            
            if action:
                tool_name = action.get('tool')
                parameters = action.get('parameters', {})
                
                action_output = f"**🔧 Action Detected:**\n"
                action_output += f"- Tool: `{tool_name}`\n"
                action_output += f"- Parameters: `{json.dumps(parameters, indent=2)}`\n\n"
                
                steps_output.append(action_output)
                
                # Execute the tool
                observation = self._execute_tool(tool_name, parameters)
                
                observation_output = f"**👁️ Observation:**\n```json\n{observation}\n```\n\n---\n"
                steps_output.append(observation_output)
                
                # Add assistant message and observation to conversation
                messages.append(AssistantMessage(content=response))
                messages.append(UserMessage(content=f"Observation: {observation}\n\nContinue reasoning or provide a Final Answer."))
                
                yield (
                    "".join(steps_output),
                    final_output
                )
            else:
                # No valid action found, prompt the agent to take action or give final answer
                no_action_output = f"⚠️ **No valid action found. Prompting agent...**\n\n---\n"
                steps_output.append(no_action_output)
                
                messages.append(AssistantMessage(content=response))
                messages.append(UserMessage(content="Please provide either a valid action in JSON format or a Final Answer."))
                
                yield (
                    "".join(steps_output),
                    final_output
                )
        
        # Max iterations reached
        final_output = f"## ⚠️ Maximum Iterations Reached\n\nI've reached the maximum number of reasoning steps ({self.max_iterations}).\n\n"
        if steps_output:
            final_output += "Please review the steps above for the investigation process."
        else:
            final_output += "I was unable to find sufficient information to answer your question."
        
        yield (
            "".join(steps_output),
            final_output
        )


# Default Adoptium repositories
DEFAULT_REPOSITORIES = [
    "adoptium/aqa-tests",
    "adoptium/TKG",
    "adoptium/aqa-systemtest",
    "adoptium/aqa-test-tools",
    "adoptium/STF",
    "adoptium/bumblebench",
    "adoptium/run-aqa",
    "adoptium/openj9-systemtest",
    "eclipse-openj9/openj9",
]

# Global agent instance
agent = ReActAgentWithUI(repositories=DEFAULT_REPOSITORIES, max_iterations=50)


def process_query(query: str):
    """Process query and stream results."""
    if not query.strip():
        return "Please enter a query.", ""
    
    # Return generator for gradio to stream
    for steps, final_answer in agent.query_with_steps(query):
        yield steps, final_answer


# Create Gradio interface
with gr.Blocks(title="VitAI ReAct Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 VitAI ReAct Agent
    
    This agent uses the **ReAct (Reasoning + Acting)** pattern to answer questions about Adoptium repositories.
    Watch as it thinks through the problem, searches GitHub, and formulates an answer!
    
    ### How it works:
    1. **Thought** - The agent reasons about what information it needs
    2. **Action** - It uses tools (search_code, search_issues) to gather information
    3. **Observation** - It analyzes the results
    4. **Repeat** - It continues until it has enough information to answer
    """)
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What testing frameworks are used in the Adoptium projects?",
                lines=3
            )
            submit_btn = gr.Button("🚀 Ask VitAI", variant="primary", size="lg")
            
            gr.Markdown("""
            ### Example Questions:
            - What testing frameworks are used in the Adoptium projects?
            - How does the TKG test harness work?
            - What are the latest issues in the aqa-tests repository?
            - Show me examples of JUnit tests in the codebase
            """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🔍 ReAct Process Steps")
            steps_output = gr.Markdown(label="Process Steps", value="")
        
        with gr.Column():
            gr.Markdown("## 📋 Final Answer")
            final_output = gr.Markdown(label="Final Answer", value="")
    
    # Set up event handler
    submit_btn.click(
        fn=process_query,
        inputs=[query_input],
        outputs=[steps_output, final_output]
    )
    
    query_input.submit(
        fn=process_query,
        inputs=[query_input],
        outputs=[steps_output, final_output]
    )


if __name__ == "__main__":
    print("🚀 Starting VitAI ReAct Agent UI...")
    print("📝 Note: Make sure you have GITHUB_TOKEN set in your .env file")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
