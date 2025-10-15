import os
import json
from typing import List, Dict, Any, Optional
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from search import GitHubSearchClient

load_dotenv()


class ReActAgent:
    """
    ReAct (Reasoning + Acting) agent that uses GitHub search tools.
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
        self.model_name = "mistral-ai/Codestral-2501"
        self.max_iterations = max_iterations
        self.repositories = repositories or []
        
        # Initialize the LLM client
        self.llm_client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.token),
        )
        
        # Initialize the GitHub search client
        self.search_client = GitHubSearchClient(token=self.token)
        
        # Cache directory structures for all repositories
        self.repo_structures: Dict[str, Dict[str, Any]] = {}
        self._load_repository_structures()
        
        # System prompt for ReAct pattern
        self.system_prompt = """You are a powerful AI agent that uses the ReAct (Reasoning + Acting) pattern to answer questions about GitHub repositories. You are autonomous and perform all necessary actions yourself.

You have access to the following tools:
1. search_code: Search for code in GitHub repositories
   - Parameters: query (str), repos (list of repo names in format owner/repo)
   - Returns: Code search results with file paths, repository names, and code snippets
   
2. search_issues: Search for issues in GitHub repositories
   - Parameters: query (str), repos (list of repo names in format owner/repo)
   - Returns: Issue search results with titles, descriptions, and metadata

3. get_repo_structure: Get the complete directory and file structure of a repository
   - Parameters: repo (str in format owner/repo), branch (optional str, defaults to default branch)
   - Returns: Complete tree structure with all directories and files in the repository
   - Use this when you need to understand the repository layout, find specific files, or see what directories exist

4. get_file_contents: Fetch the actual contents of a specific file from a repository
   - Parameters: repo (str in format owner/repo), path (str, file path in repo), branch (optional str)
   - Returns: File contents with metadata (content is base64 encoded for binary files, decoded for text files)
   - CRITICAL: Use this tool when you need to read the actual content of a file, not just find it
   - You MUST call this API to get file contents - never assume or make up file contents

REPOSITORY CONTEXT:
You will be provided with a detailed repository structure context showing:
- Directories that exist in each repository
- File types available (extensions)
- Key files in the repository root

Use this context to formulate ACCURATE search queries with proper qualifiers.

Follow the GitHub search syntax rules:
- Use qualifiers like 'language:', 'extension:', 'path:', 'filename:' for code search
- Use qualifiers like 'is:issue', 'is:pr', 'state:open', 'label:' for issue search
- Combine terms with spaces for AND logic, use OR for alternatives
- IMPORTANT: Use the repository structure context to craft precise queries that will return results

CRITICAL: Use the ReAct pattern to solve problems. You MUST follow this exact format:

Thought: [Explain your reasoning about what information you need and which tool to use]

Action:
{"tool": "search_code", "parameters": {"query": "your search query", "repos": ["owner/repo"]}}

After you receive an Observation, you can either:
- Take another Action (same JSON format)
- Provide a Final Answer: [your complete answer]

EXAMPLES OF VALID ACTIONS:
{"tool": "search_code", "parameters": {"query": "junit language:java", "repos": ["adoptium/aqa-tests"]}}
{"tool": "search_issues", "parameters": {"query": "test framework is:issue", "repos": ["adoptium/aqa-tests"]}}
{"tool": "get_repo_structure", "parameters": {"repo": "adoptium/aqa-tests"}}
{"tool": "get_repo_structure", "parameters": {"repo": "adoptium/aqa-tests", "branch": "main"}}
{"tool": "get_file_contents", "parameters": {"repo": "adoptium/aqa-tests", "path": "build.xml"}}
{"tool": "get_file_contents", "parameters": {"repo": "adoptium/aqa-tests", "path": "src/main/Config.java", "branch": "main"}}

IMPORTANT RULES:
- Always start with a Thought
- Action MUST be valid JSON on its own line
- After receiving an Observation, continue with another Thought
- CRITICAL: Use the provided repository structure context to formulate accurate search queries
- When searching, use specific paths, extensions, and filenames based on the structure context
- Avoid generic searches - be specific based on what actually exists in the repository
- Use get_repo_structure ONLY if you need more detailed information than what's in the context
- Use search_code to find code files, dependencies, configurations
- ALWAYS use get_file_contents to read actual file contents - NEVER assume or make up file contents
- Use search_issues to find discussions, problems, or documentation
- When you find a relevant file path, immediately use get_file_contents to read it
- When you have gathered enough information, provide "Final Answer:" followed by your answer
- Be specific in your search queries to get better results

YOUR MISSION:
- YOU are the one performing all actions - the user is simply waiting for your answer
- NEVER tell the user to perform any action themselves (e.g., "you can search for...", "try checking...", "you should look at...")
- NEVER suggest the user do something - YOU must do it yourself using your tools
- Always provide COMPLETE, ACTIONABLE solutions based on the information you gather
- If you need more information, use your tools to get it - don't ask the user to find it
- Your Final Answer must be a complete solution, not a list of things for the user to do
- Be thorough and persistent - use multiple searches if needed to gather all necessary information
- The user expects YOU to solve their problem completely, not to receive instructions on how they should solve it
- CRITICAL: When you need file contents, you MUST call get_file_contents API - never assume what's in a file"""

    def _load_repository_structures(self):
        """
        Load and cache directory structures for all repositories.
        This provides context for better search queries.
        """
        print(f"Loading repository structures for {len(self.repositories)} repositories...")
        
        for repo in self.repositories:
            try:
                print(f"  Loading structure for {repo}...", end=" ")
                owner, repo_name = repo.split('/', 1)
                
                # Get repository structure
                tree_data = self.search_client.get_repository_tree(
                    owner=owner,
                    repo=repo_name,
                    branch=None,
                    recursive=True
                )
                
                # Organize the tree into a more readable structure
                tree_items = tree_data.get('tree', [])
                directories = []
                files_by_type = {}
                
                for item in tree_items:
                    path = item.get('path', '')
                    item_type = item.get('type')
                    
                    if item_type == 'tree':
                        directories.append(path)
                    else:
                        # Categorize files by extension
                        if '.' in path:
                            ext = path.split('.')[-1].lower()
                            if ext not in files_by_type:
                                files_by_type[ext] = []
                            files_by_type[ext].append(path)
                        else:
                            if 'no_extension' not in files_by_type:
                                files_by_type['no_extension'] = []
                            files_by_type['no_extension'].append(path)
                
                self.repo_structures[repo] = {
                    'total_items': len(tree_items),
                    'total_directories': len(directories),
                    'directories': sorted(directories),
                    'files_by_extension': {ext: sorted(files) for ext, files in files_by_type.items()},
                    'file_extensions': sorted(files_by_type.keys())
                }
                
                print(f"✓ ({len(tree_items)} items)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                self.repo_structures[repo] = {'error': str(e)}
        
        print("Repository structures loaded.\n")
    
    def _get_repo_context_summary(self) -> str:
        """
        Generate a summary of repository structures for the LLM context.
        This helps the LLM formulate better search queries.
        """
        if not self.repo_structures:
            return "No repository structure information available."
        
        summary_parts = ["REPOSITORY STRUCTURE CONTEXT:"]
        
        for repo, structure in self.repo_structures.items():
            if 'error' in structure:
                summary_parts.append(f"\n{repo}: [Error loading structure]")
                continue
            
            summary_parts.append(f"\n{repo}:")
            summary_parts.append(f"  Total files/dirs: {structure['total_items']}")
            
            # Show key directories (limit to important ones)
            key_dirs = [d for d in structure['directories'][:20] 
                       if not d.startswith('.') and '/' not in d[1:]]
            if key_dirs:
                summary_parts.append(f"  Top-level directories: {', '.join(key_dirs[:10])}")
            
            # Show file types available
            extensions = structure['file_extensions'][:15]
            if extensions:
                summary_parts.append(f"  File types: {', '.join(extensions)}")
            
            # Show specific important files
            important_files = []
            for ext in ['xml', 'gradle', 'java', 'py', 'md', 'json', 'yaml', 'yml']:
                if ext in structure['files_by_extension']:
                    files = structure['files_by_extension'][ext]
                    # Show root-level files
                    root_files = [f for f in files if '/' not in f]
                    if root_files:
                        important_files.extend(root_files[:3])
            
            if important_files:
                summary_parts.append(f"  Key files: {', '.join(important_files[:5])}")
        
        return '\n'.join(summary_parts)

    def _call_llm(self, messages: List[Any]) -> str:
        """Call the LLM with the given messages."""
        response = self.llm_client.complete(
            messages=messages,
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,  # Reduced to leave room for context
            model=self.model_name
        )
        return response.choices[0].message.content

    def _truncate_observation(self, observation: str, max_length: int = 1500) -> str:
        """Truncate long observations to prevent context overflow."""
        if len(observation) <= max_length:
            return observation
        
        # Try to parse as JSON and truncate items
        try:
            data = json.loads(observation)
            if 'items' in data and isinstance(data['items'], list):
                original_count = len(data['items'])
                # Keep only first 3 items
                data['items'] = data['items'][:3]
                if original_count > 3:
                    data['note'] = f"Showing 3 of {original_count} total items to save context"
                return json.dumps(data, indent=2)
        except:
            pass
        
        # Fallback: simple truncation
        return observation[:max_length] + f"\n... [truncated, originally {len(observation)} chars]"

    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        Execute the specified tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute ('search_code', 'search_issues', 'get_repo_structure', or 'get_file_contents')
            parameters: Parameters for the tool
            
        Returns:
            JSON string containing the results
        """
        try:
            if tool_name == 'get_file_contents':
                # Parse parameters
                repo = parameters.get('repo', '')
                path = parameters.get('path', '')
                branch = parameters.get('branch', None)
                
                if not repo or '/' not in repo:
                    return json.dumps({'error': 'Invalid repo parameter. Must be in format owner/repo'})
                
                if not path:
                    return json.dumps({'error': 'path parameter is required'})
                
                owner, repo_name = repo.split('/', 1)
                
                # Get file contents from GitHub API
                file_data = self.search_client.get_file_contents(
                    owner=owner,
                    repo=repo_name,
                    path=path,
                    branch=branch
                )
                
                # Decode content if it's a text file
                import base64
                content = file_data.get('content', '')
                encoding = file_data.get('encoding', '')
                
                decoded_content = None
                if encoding == 'base64' and content:
                    try:
                        decoded_bytes = base64.b64decode(content)
                        # Try to decode as UTF-8 text
                        try:
                            decoded_content = decoded_bytes.decode('utf-8')
                        except UnicodeDecodeError:
                            decoded_content = f"[Binary file, size: {len(decoded_bytes)} bytes]"
                    except Exception as e:
                        decoded_content = f"[Error decoding content: {str(e)}]"
                
                return json.dumps({
                    'repository': repo,
                    'path': path,
                    'name': file_data.get('name'),
                    'size': file_data.get('size'),
                    'type': file_data.get('type'),
                    'sha': file_data.get('sha'),
                    'encoding': encoding,
                    'content': decoded_content or content,
                    'html_url': file_data.get('html_url')
                }, indent=2)
            
            if tool_name == 'get_repo_structure':
                # Parse repo parameter (format: owner/repo)
                repo = parameters.get('repo', '')
                branch = parameters.get('branch', None)
                
                if not repo or '/' not in repo:
                    return json.dumps({'error': 'Invalid repo parameter. Must be in format owner/repo'})
                
                owner, repo_name = repo.split('/', 1)
                
                # Get repository structure
                tree_data = self.search_client.get_repository_tree(
                    owner=owner,
                    repo=repo_name,
                    branch=branch,
                    recursive=True
                )
                
                # Organize the tree into a more readable structure
                tree_items = tree_data.get('tree', [])
                directories = []
                files = []
                
                for item in tree_items:
                    item_info = {
                        'path': item.get('path'),
                        'type': item.get('type'),
                        'size': item.get('size')
                    }
                    if item.get('type') == 'tree':
                        directories.append(item_info['path'])
                    else:
                        files.append(item_info)
                
                return json.dumps({
                    'repository': repo,
                    'total_items': len(tree_items),
                    'total_directories': len(directories),
                    'total_files': len(files),
                    'directories': sorted(directories),
                    'files': sorted([f['path'] for f in files])
                }, indent=2)
            
            repos = parameters.get('repos', self.repositories)
            query = parameters.get('query', '')
            
            # Build the full query string with repo qualifiers
            # GitHub search requires repo qualifiers to be part of the query string
            full_query = query
            if repos:
                # Add each repo as a separate qualifier in the query
                repo_qualifiers = ' '.join(f'repo:{repo}' for repo in repos)
                full_query = f"{query} {repo_qualifiers}".strip()
            
            if tool_name == 'search_code':
                results = self.search_client.search_code(
                    q=full_query,
                    qualifiers=None,  # Already included in the query string
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
                    q=full_query,
                    qualifiers=None,  # Already included in the query string
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
        Handles both inline JSON and JSON on separate lines after "Action:"
        
        IMPORTANT: Does not parse JSON that appears after "Final Answer:"
        """
        # Don't parse actions if we have a final answer
        if "Final Answer:" in text or "final answer:" in text.lower():
            return None
        
        try:
            # Method 1: Look for JSON after "Action:" keyword
            if "Action:" in text:
                action_idx = text.find("Action:")
                text_after_action = text[action_idx + 7:].strip()
                # Find the first { after Action:
                start_idx = text_after_action.find('{')
                if start_idx != -1:
                    # Find matching closing brace
                    brace_count = 0
                    for i, char in enumerate(text_after_action[start_idx:]):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = text_after_action[start_idx:start_idx + i + 1]
                                action = json.loads(json_str)
                                if 'tool' in action and 'parameters' in action:
                                    return action
                                break
            
            # Method 2: Try to find any JSON in the text
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx + 1]
                action = json.loads(json_str)
                
                if 'tool' in action and 'parameters' in action:
                    return action
            
            return None
        except (json.JSONDecodeError, Exception) as e:
            # Try to find multiple JSON objects and parse them
            import re
            json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}'
            matches = re.findall(json_pattern, text)
            for match in matches:
                try:
                    action = json.loads(match)
                    if 'tool' in action and 'parameters' in action:
                        return action
                except:
                    continue
            return None

    def query(self, user_query: str, repositories: Optional[List[str]] = None) -> str:
        """
        Process a user query using the ReAct pattern.
        
        Args:
            user_query: The user's question
            repositories: Optional list of repositories to override the default
            
        Returns:
            The agent's final answer
        """
        # Use provided repositories or fall back to instance repositories
        if repositories:
            self.repositories = repositories
            # Reload repository structures if repositories changed
            self._load_repository_structures()
        
        # Get repository context summary
        repo_context = self._get_repo_context_summary()
        
        # Initialize conversation with system prompt and user query
        messages = [
            SystemMessage(content=self.system_prompt),
            UserMessage(content=f"""Question: {user_query}

Available repositories: {', '.join(self.repositories)}

{repo_context}

IMPORTANT: Use the repository structure context above to formulate accurate search queries.
- Use 'path:' qualifier to search in specific directories you see above
- Use 'extension:' or 'language:' qualifiers to search specific file types
- Use 'filename:' to search for specific files you see in the structure
- Be specific with your search terms based on what exists in the repository

Please start with your Thought and then provide an Action in valid JSON format.""")
        ]
        
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Get response from LLM
            response = self._call_llm(messages)
            
            # Check if we have a final answer FIRST (before trying to parse actions)
            if "Final Answer:" in response or "final answer:" in response.lower():
                # Extract the final answer
                if "Final Answer:" in response:
                    final_answer = response.split("Final Answer:")[-1].strip()
                else:
                    # Case-insensitive search
                    idx = response.lower().find("final answer:")
                    final_answer = response[idx + len("final answer:"):].strip()
                return final_answer
            
            # Only try to parse an action if we don't have a final answer
            action = self._parse_action(response)
            
            if action:
                tool_name = action.get('tool')
                parameters = action.get('parameters', {})
                
                # Execute the tool
                observation = self._execute_tool(tool_name, parameters)
                
                # Truncate observation to prevent context overflow
                observation = self._truncate_observation(observation)
                
                # Add assistant message and observation to conversation
                messages.append(AssistantMessage(content=response))
                messages.append(UserMessage(content=f"Observation: {observation}\n\nBased on this observation, provide either:\n1. Another Thought and Action to gather more information\n2. Final Answer: [your complete answer if you have enough information]"))
            else:
                # No valid action found and no final answer, provide clearer guidance
                messages.append(AssistantMessage(content=response))
                messages.append(UserMessage(content=f"I didn't find a valid action in your response. Please provide EITHER:\n\n1. A JSON action in one of these formats:\n\n   For searching code:\n   {{\"tool\": \"search_code\", \"parameters\": {{\"query\": \"your search query\", \"repos\": {self.repositories}}}}}\n\n   For searching issues:\n   {{\"tool\": \"search_issues\", \"parameters\": {{\"query\": \"your search query\", \"repos\": {self.repositories}}}}}\n\n   For getting repository structure:\n   {{\"tool\": \"get_repo_structure\", \"parameters\": {{\"repo\": \"owner/repo\"}}}}\n\nOR\n\n2. Final Answer: [your answer if you have enough information]\n\nRemember: Start with 'Thought:' to explain your reasoning, then provide 'Action:' with the JSON."))
        
        # Max iterations reached - provide the best answer we can
        return f"I've reached the maximum number of reasoning steps ({self.max_iterations}). Based on my investigation, I was unable to gather sufficient information to provide a complete answer to your question. Please try rephrasing your question or being more specific about what you'd like to know."


# Factory function for creating agent instance
def create_agent(repositories: Optional[List[str]] = None, max_iterations: int = 5) -> ReActAgent:
    """
    Create a ReAct agent instance.
    
    Args:
        repositories: List of repositories to search
        max_iterations: Maximum reasoning iterations
        
    Returns:
        Configured ReActAgent instance
    """
    return ReActAgent(repositories=repositories, max_iterations=max_iterations)

