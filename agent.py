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
        
        # Cache directory structures for all repositories at initialization
        # This provides the agent with immediate navigation context
        self.repo_structures: Dict[str, Dict[str, Any]] = {}
        self.repo_structures_text: Dict[str, str] = {}  # Store full text structures
        if self.repositories:
            print(f"\n🔄 Initializing agent with {len(self.repositories)} repositories...")
            self._load_repository_structures()
            print("✅ Agent initialization complete - repository structures cached and ready!\n")
        else:
            print("⚠️ Warning: No repositories specified. Repository structures will be empty.")
        
        # System prompt for ReAct pattern
        self.system_prompt = """You are a powerful AI agent that uses the ReAct (Reasoning + Acting) pattern to answer questions about GitHub repositories. You are an AUTONOMOUS EXPERT that performs deep code exploration and analysis to provide COMPLETE, ACTIONABLE answers.

🎯 YOUR CORE MISSION:
- YOU are the expert doing ALL the investigative work
- The user is NOT a developer - they cannot search code, read files, or analyze repositories
- NEVER tell the user what to do, where to look, or what to search
- ALWAYS do the work yourself and provide THE ACTUAL ANSWER, not directions to find it
- Think of yourself as a senior developer who has been asked a question - you investigate thoroughly and come back with the complete solution

⚠️ CRITICAL: ONE STEP AT A TIME
- You operate in an INTERACTIVE loop
- Provide ONE Thought and ONE Action, then STOP
- The system will execute your action and give you the real Observation
- NEVER simulate multiple steps in one response
- NEVER write "Step 1, Step 2, Step 3..." in a single response
- NEVER fabricate observations or results
- NEVER provide "Final Answer" until you've actually completed your investigation

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
You will be provided with COMPLETE SAVED REPOSITORY STRUCTURES from disk that include:
- Full hierarchical directory tree of each repository
- Complete file listings with sizes
- File types and extensions with counts
- Root-level configuration files
- Path information for every file and directory

These structures are PRE-LOADED from saved text files, giving you immediate and comprehensive access
to the entire repository layout WITHOUT needing to call get_repo_structure API.

USE THIS SAVED STRUCTURE DATA TO:
✓ Identify exact file paths before calling get_file_contents
✓ Navigate directly to relevant directories and files
✓ Understand the project organization and architecture
✓ Formulate precise search queries with accurate path: qualifiers
✓ Know exactly which files exist before attempting to read them

Follow the GitHub search syntax rules:
- Use qualifiers like 'language:', 'extension:', 'path:', 'filename:' for code search
- Use qualifiers like 'is:issue', 'is:pr', 'state:open', 'label:' for issue search
- Combine terms with spaces for AND logic, use OR for alternatives
- IMPORTANT: Use the repository structure context to craft precise queries that will return results

CRITICAL: Use the ReAct pattern to solve problems. You MUST follow this exact format:

Thought: [Explain your reasoning about what information you need and which tool to use]

Action:
{"tool": "search_code", "parameters": {"query": "your search query", "repos": ["owner/repo"]}}

⚠️ CRITICAL RULES FOR YOUR RESPONSES:
1. Provide ONLY ONE Thought and ONE Action per response
2. NEVER simulate or predict what the Observation will be
3. NEVER write out multiple steps (Step 1, Step 2, etc.) in a single response
4. NEVER include "Final Answer:" until you have actually executed all necessary actions and received observations
5. Wait for the actual Observation from the system after each action
6. Do NOT fabricate or imagine observations - you will receive real ones after each action

❌ WRONG (simulating entire conversation):
Thought: I'll search for tests
Action: {...}
Observation: [imagined result]
Step 2: ...
Final Answer: [premature answer]

✅ CORRECT (one step at a time):
Thought: I'll search for tests
Action: {...}
[STOP - wait for real observation]

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

🎯 RECOMMENDED WORKFLOW WITH SAVED STRUCTURES:
1. READ the repository structure provided in context (already loaded from saved files)
2. IDENTIFY exact file paths of interest from the structure
3. USE get_file_contents to read those specific files
4. ANALYZE the actual content you receive
5. PROVIDE your final answer based on real observations

Example - Finding build configuration:
  Thought: The saved structure shows build.xml exists at the root. I'll read it to find build configuration details.
  Action: {"tool": "get_file_contents", "parameters": {"repo": "adoptium/aqa-tests", "path": "build.xml"}}
  [Wait for observation with actual file contents]
  [After receiving the content, analyze it and provide answer]

Example - Finding test files:
  Thought: The structure shows test files in src/test/ directories. Let me search for JUnit test files.
  Action: {"tool": "search_code", "parameters": {"query": "junit path:src/test extension:java", "repos": ["adoptium/aqa-tests"]}}
  [Wait for observation]
  Thought: Now I'll read one of the test files to see how tests are structured.
  Action: {"tool": "get_file_contents", "parameters": {"repo": "adoptium/aqa-tests", "path": "src/test/java/TestExample.java"}}
  [Wait for observation and then provide answer]
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

🚫 ABSOLUTELY FORBIDDEN - NEVER DO THESE:
❌ "You can check the X directory..."
❌ "You should look at the Y file..."
❌ "Try searching for Z..."
❌ "The code might be in..."
❌ "You could investigate..."
❌ "I suggest examining..."
❌ "Based on the structure, you can find..."
❌ Providing file paths or directories for the user to explore
❌ Suggesting searches the user should perform
❌ Giving the user a roadmap or investigation plan
❌ Saying "here's where you should look" or similar phrases

✅ REQUIRED BEHAVIOR - ALWAYS DO THIS:
✓ Search for files yourself using search_code
✓ Read file contents yourself using get_file_contents
✓ Analyze the actual code and provide concrete findings
✓ State "I found that..." or "After examining the code, I discovered..."
✓ Provide specific answers like "The test framework used is JUnit 5, which I found in pom.xml at line 45"
✓ Give complete, ready-to-use information extracted from the actual code
✓ If asked about dependencies, list the ACTUAL dependencies you found
✓ If asked about structure, describe what YOU discovered after exploring
✓ If asked about implementation, show the ACTUAL code you found

YOUR MISSION - WHAT MAKES A GOOD FINAL ANSWER:
🎯 GOOD Final Answer: "I examined the repository and found that the project uses JUnit 5 for testing. The dependency is declared in pom.xml with version 5.8.2. The main test directory is located at src/test/java, and I found 45 test files. The tests use annotations like @Test, @BeforeEach, and @ParameterizedTest. Here's an example from UserServiceTest.java: [actual code snippet]"

❌ BAD Final Answer: "You can find the testing framework by looking at the pom.xml file in the root directory. The test files are in the src/test/java directory. You should check the @Test annotations to understand how tests are structured."

KEY PRINCIPLE: Treat the user as a non-technical stakeholder who hired you to investigate the codebase. They cannot and will not do any technical investigation themselves. Your job is to dive deep, explore thoroughly, and come back with THE COMPLETE ANSWER, not a treasure map."""

    def _load_repository_structures(self):
        """
        Load and cache directory structures from saved text files.
        This is called during initialization to provide immediate navigation context.
        The saved structures help the agent formulate precise search queries and navigate efficiently.
        
        Priority:
        1. Try to load from saved files in repo_structures/ directory
        2. If not found, fall back to GitHub API (and optionally save to file)
        """
        print(f"📂 Loading repository structures from saved files...")
        
        repo_structures_dir = os.path.join(os.path.dirname(__file__), 'repo_structures')
        
        for idx, repo in enumerate(self.repositories, 1):
            try:
                print(f"   [{idx}/{len(self.repositories)}] Loading {repo}...", end=" ", flush=True)
                
                # Convert repo name to filename format (owner/repo -> owner_repo.txt)
                safe_repo_name = repo.replace('/', '_')
                structure_file = os.path.join(repo_structures_dir, f"{safe_repo_name}.txt")
                
                # Try to load from saved file first
                if os.path.exists(structure_file):
                    with open(structure_file, 'r', encoding='utf-8') as f:
                        full_text = f.read()
                        self.repo_structures_text[repo] = full_text
                    
                    # Parse the file to extract structured data for quick access
                    directories = []
                    files_by_type = {}
                    
                    # Parse directory section
                    if "All directories (hierarchical):" in full_text:
                        dir_section = full_text.split("All directories (hierarchical):")[1].split("\n\n")[0]
                        for line in dir_section.split('\n'):
                            if '📁' in line:
                                # Extract directory path (remove emoji and whitespace)
                                dir_name = line.split('📁')[1].strip().rstrip('/')
                                if dir_name:
                                    directories.append(dir_name)
                    
                    # Parse file types summary
                    if "FILE TYPES SUMMARY" in full_text:
                        types_section = full_text.split("FILE TYPES SUMMARY")[1].split("=" * 80)[0]
                        for line in types_section.split('\n'):
                            if line.strip().startswith('.'):
                                parts = line.strip().split(':')
                                if len(parts) == 2:
                                    ext = parts[0].strip('.').strip()
                                    if ext == '[no extension]':
                                        ext = 'no_extension'
                                    files_by_type[ext] = []
                    
                    # Store minimal structured data
                    self.repo_structures[repo] = {
                        'loaded_from_file': True,
                        'file_path': structure_file,
                        'directories': directories,
                        'file_extensions': list(files_by_type.keys()),
                        'has_full_text': True
                    }
                    
                    print(f"✓ Loaded from {safe_repo_name}.txt")
                else:
                    # Fallback: load from GitHub API
                    print(f"⚠️  File not found, loading from API...", end=" ", flush=True)
                    owner, repo_name = repo.split('/', 1)
                    
                    tree_data = self.search_client.get_repository_tree(
                        owner=owner,
                        repo=repo_name,
                        branch=None,
                        recursive=True
                    )
                    
                    # Organize the tree into a structured format
                    tree_items = tree_data.get('tree', [])
                    directories = []
                    files_by_type = {}
                    
                    for item in tree_items:
                        path = item.get('path', '')
                        item_type = item.get('type')
                        
                        if item_type == 'tree':
                            directories.append(path)
                        else:
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
                        'loaded_from_file': False,
                        'total_items': len(tree_items),
                        'total_directories': len(directories),
                        'directories': sorted(directories),
                        'files_by_extension': {ext: sorted(files) for ext, files in files_by_type.items()},
                        'file_extensions': sorted(files_by_type.keys())
                    }
                    
                    total_files = len(tree_items) - len(directories)
                    print(f"✓ ({total_files} files, {len(directories)} dirs)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                self.repo_structures[repo] = {'error': str(e)}
        
        loaded_from_file = len([r for r in self.repo_structures.values() if r.get('loaded_from_file')])
        loaded_from_api = len([r for r in self.repo_structures.values() if r.get('loaded_from_file') == False])
        print(f"✅ Successfully loaded {loaded_from_file} from files, {loaded_from_api} from API\n")
    
    def _get_repo_context_summary(self) -> str:
        """
        Generate a comprehensive summary of repository structures for the LLM context.
        If full text structures are available from saved files, use those.
        Otherwise, generate from the structured data.
        """
        if not self.repo_structures:
            return "⚠️ No repository structure information available. Use get_repo_structure tool to load it."
        
        summary_parts = ["=" * 80]
        summary_parts.append("📁 COMPLETE REPOSITORY STRUCTURES - LOADED FROM SAVED FILES")
        summary_parts.append("=" * 80)
        summary_parts.append("\n🎯 THESE ARE COMPLETE, COMPREHENSIVE REPOSITORY STRUCTURES")
        summary_parts.append("You have access to:")
        summary_parts.append("  • Full hierarchical directory trees")
        summary_parts.append("  • Complete file listings with exact paths")
        summary_parts.append("  • File sizes and types")
        summary_parts.append("  • Root-level configuration files")
        summary_parts.append("  • All subdirectories and their contents")
        summary_parts.append("")
        summary_parts.append("💡 HOW TO USE THIS:")
        summary_parts.append("  1. READ the full structure below to understand the repository layout")
        summary_parts.append("  2. IDENTIFY exact file paths you need to examine")
        summary_parts.append("  3. USE get_file_contents with the exact paths to read files")
        summary_parts.append("  4. FORMULATE precise search queries using path:, filename:, extension: qualifiers")
        summary_parts.append("  5. PROVIDE complete answers based on actual file contents you retrieve")
        summary_parts.append("")
        
        for repo, structure in self.repo_structures.items():
            if 'error' in structure:
                summary_parts.append(f"\n❌ {repo}: [Error loading structure: {structure['error']}]")
                summary_parts.append("   Use get_repo_structure tool if you need to retry loading this repo.")
                continue
            
            # If we have full text from saved file, include a relevant portion
            if structure.get('has_full_text') and repo in self.repo_structures_text:
                full_text = self.repo_structures_text[repo]
                
                # Extract key sections for context (limit to avoid token overflow)
                summary_parts.append(f"\n{'=' * 80}")
                summary_parts.append(f"📦 Repository: {repo}")
                summary_parts.append(f"{'=' * 80}")
                summary_parts.append(f"✓ Loaded from saved file: {structure.get('file_path', 'N/A')}")
                summary_parts.append("")
                
                # Include the summary section from the file (first ~150 lines or until a delimiter)
                lines = full_text.split('\n')
                include_lines = []
                in_summary = True
                line_count = 0
                
                for line in lines:
                    line_count += 1
                    
                    # Include header and summary sections
                    if line_count < 150:
                        include_lines.append(line)
                    # Stop at hierarchical tree view to save tokens
                    elif "HIERARCHICAL TREE VIEW" in line:
                        include_lines.append("\n[... Full hierarchical tree and detailed file listings available ...]")
                        include_lines.append(f"[... See {structure.get('file_path')} for complete details ...]")
                        break
                
                summary_parts.append('\n'.join(include_lines[:200]))  # Limit to 200 lines max
                summary_parts.append("")
                
            else:
                # Fallback: use structured data (legacy format)
                summary_parts.append(f"\n{'=' * 80}")
                summary_parts.append(f"📦 Repository: {repo}")
                summary_parts.append(f"{'=' * 80}")
                
                if 'total_items' in structure:
                    summary_parts.append(f"📊 Total items: {structure['total_items']} ({structure.get('total_directories', 0)} directories, {structure['total_items'] - structure.get('total_directories', 0)} files)")
                
                # Show top-level directory structure
                directories = structure.get('directories', [])
                top_level_dirs = [d for d in directories if '/' not in d and not d.startswith('.')]
                if top_level_dirs:
                    summary_parts.append(f"\n📂 Top-level directories ({len(top_level_dirs)}):")
                    summary_parts.append(f"   {', '.join(sorted(top_level_dirs[:15]))}")
                    if len(top_level_dirs) > 15:
                        summary_parts.append(f"   ... and {len(top_level_dirs) - 15} more")
                
                # Show file extensions
                extensions = structure.get('file_extensions', [])
                if extensions:
                    summary_parts.append(f"\n📄 Available file types ({len(extensions)} types):")
                    summary_parts.append(f"   {', '.join([f'.{ext}' for ext in extensions[:20]])}")
                    if len(extensions) > 20:
                        summary_parts.append(f"   ... and {len(extensions) - 20} more types")
        
        summary_parts.append(f"\n{'=' * 80}")
        summary_parts.append("🔍 IMPORTANT: USE get_file_contents TO READ ACTUAL FILE CONTENTS")
        summary_parts.append("=" * 80)
        summary_parts.append("The structures above show you WHAT files exist and WHERE they are.")
        summary_parts.append("To answer questions, you MUST:")
        summary_parts.append("  1. Identify relevant files from the structure above")
        summary_parts.append("  2. Use get_file_contents to read those files")
        summary_parts.append("  3. Analyze the actual content")
        summary_parts.append("  4. Provide answers based on what you actually found")
        summary_parts.append("")
        summary_parts.append("Example workflow:")
        summary_parts.append('  Thought: I need to check the build configuration')
        summary_parts.append('  Action: {"tool": "get_file_contents", "parameters": {"repo": "owner/repo", "path": "build.xml"}}')
        summary_parts.append("  [Wait for observation with actual file contents]")
        summary_parts.append("  [Analyze the contents and provide answer]")
        summary_parts.append(f"{'=' * 80}\n")
        
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
        
        Returns the action even if "Final Answer" is mentioned, as long as there's a valid action JSON.
        The _extract_final_answer method will determine if it's truly a final answer.
        """
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
            
            # Method 2: Try to find any JSON in the text (before any "Final Answer:" marker)
            # Split at "Final Answer:" to avoid parsing JSON examples in the final answer
            search_text = text
            if "Final Answer:" in text:
                search_text = text.split("Final Answer:")[0]
            elif "final answer:" in text.lower():
                idx = text.lower().find("final answer:")
                search_text = text[:idx]
            
            start_idx = search_text.find('{')
            end_idx = search_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = search_text[start_idx:end_idx + 1]
                action = json.loads(json_str)
                
                if 'tool' in action and 'parameters' in action:
                    return action
            
            return None
        except (json.JSONDecodeError, Exception) as e:
            # Try to find multiple JSON objects and parse them
            import re
            
            # Only search before "Final Answer:" marker
            search_text = text
            if "Final Answer:" in text:
                search_text = text.split("Final Answer:")[0]
            elif "final answer:" in text.lower():
                idx = text.lower().find("final answer:")
                search_text = text[:idx]
            
            json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}'
            matches = re.findall(json_pattern, search_text)
            for match in matches:
                try:
                    action = json.loads(match)
                    if 'tool' in action and 'parameters' in action:
                        return action
                except:
                    continue
            return None

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """
        Extract final answer from the agent's response.
        Only returns a final answer if it's clearly marked and not just mentioned in passing.
        
        Rules for detecting a final answer:
        1. "Final Answer:" must appear at the start of a line (after optional whitespace)
        2. Must be followed by actual content (not empty)
        3. Should not have an "Action:" or "Observation:" after it (agent is simulating)
        4. Should not have "Step 1", "Step 2" patterns (agent is planning multiple steps)
        
        Returns:
            The final answer text, or None if no valid final answer found
        """
        # Detect if agent is simulating a multi-step conversation
        # Common patterns: "Step 1:", "Step 2:", multiple "Thought:", multiple "Action:", "Observation:" in text
        simulation_patterns = [
            'Step 1:',
            'Step 2:',
            'Step 3:',
            '### Step',
            '## Step',
        ]
        
        # Count how many times these appear
        thought_count = text.count('Thought:')
        action_count = text.count('Action:')
        observation_in_text = 'Observation:' in text  # Agent shouldn't write "Observation:" - system does
        
        # If agent is simulating multiple steps, reject all final answers
        has_simulation_pattern = any(pattern in text for pattern in simulation_patterns)
        if has_simulation_pattern or observation_in_text or thought_count > 1 or action_count > 1:
            return None
        
        lines = text.split('\n')
        
        final_answer_line_idx = -1
        action_line_idx = -1
        
        # Find the last occurrence of "Final Answer:" at line start
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("Final Answer:") or stripped.lower().startswith("final answer:"):
                final_answer_line_idx = idx
            if "Action:" in line and '{' in text[text.find(line):]:
                action_line_idx = idx
        
        # If there's an Action after Final Answer mention, it's not a final answer yet
        if action_line_idx > final_answer_line_idx and final_answer_line_idx != -1:
            return None
        
        # No Final Answer found at line start
        if final_answer_line_idx == -1:
            return None
        
        # Extract the final answer
        final_answer_line = lines[final_answer_line_idx]
        
        # Find the "Final Answer:" marker (case-insensitive)
        if "Final Answer:" in final_answer_line:
            answer = final_answer_line.split("Final Answer:", 1)[1].strip()
        else:
            # Case-insensitive
            lower_line = final_answer_line.lower()
            idx = lower_line.find("final answer:")
            answer = final_answer_line[idx + len("final answer:"):].strip()
        
        # Add any lines after the Final Answer line
        remaining_lines = lines[final_answer_line_idx + 1:]
        if remaining_lines:
            answer += "\n" + "\n".join(remaining_lines)
        
        # Must have actual content (not just whitespace)
        if answer and answer.strip():
            return answer.strip()
        
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
            # Check if we need to reload structures
            repos_changed = set(repositories) != set(self.repositories)
            self.repositories = repositories
            
            if repos_changed:
                print(f"\n🔄 Repository list changed. Reloading structures for {len(repositories)} repositories...")
                self._load_repository_structures()
        
        # Ensure we have repository structures cached
        if not self.repo_structures and self.repositories:
            print(f"\n🔄 Repository structures not cached. Loading now...")
            self._load_repository_structures()
        
        # Get repository context summary from cache
        repo_context = self._get_repo_context_summary()
        
        # Initialize conversation with system prompt and user query
        messages = [
            SystemMessage(content=self.system_prompt),
            UserMessage(content=f"""Question: {user_query}

{'=' * 80}
AVAILABLE REPOSITORIES: {', '.join(self.repositories)}
{'=' * 80}

{repo_context}

🎯 IMPORTANT INSTRUCTIONS:
The repository structures above have been LOADED FROM SAVED FILES containing complete repository data.
These are NOT summaries - they contain FULL, DETAILED information about every file and directory.

YOU HAVE COMPLETE REPOSITORY STRUCTURES AT YOUR DISPOSAL!

RECOMMENDED APPROACH:
1. 📖 READ the saved structure above carefully - it shows ALL files and directories
2. 🎯 IDENTIFY exact file paths relevant to the question (e.g., build.xml, pom.xml, specific .java files)
3. 📥 USE get_file_contents to read those specific files with exact paths
4. 🔍 If you need to find files by content, use search_code with precise path: and extension: qualifiers
5. ✅ PROVIDE answers based on actual file contents you retrieve

WHY USE SAVED STRUCTURES:
✓ Instant access to complete repository layout without API calls
✓ See exact file paths before reading them
✓ Understand project organization immediately
✓ Formulate targeted searches instead of broad ones
✓ Navigate efficiently to the right files

TOOL USAGE PRIORITY:
1️⃣ get_file_contents: When you know the exact file path from the structure (MOST COMMON)
   Example: {"tool": "get_file_contents", "parameters": {"repo": "adoptium/aqa-tests", "path": "build.xml"}}

2️⃣ search_code: When you need to find files by content or pattern
   Example: {"tool": "search_code", "parameters": {"query": "junit path:src/test extension:java", "repos": ["adoptium/aqa-tests"]}}

3️⃣ search_issues: When looking for discussions or bug reports
   Example: {"tool": "search_issues", "parameters": {"query": "test failure is:issue", "repos": ["adoptium/aqa-tests"]}}

4️⃣ get_repo_structure: RARELY NEEDED - only if saved structure is missing or you need latest updates
   Example: {"tool": "get_repo_structure", "parameters": {"repo": "adoptium/aqa-tests"}}

⚠️ CRITICAL REMINDER: ONE STEP AT A TIME!
- Provide ONLY ONE Thought and ONE Action in your response
- Do NOT write "Step 1:", "Step 2:", etc.
- Do NOT simulate observations or results
- Do NOT write multiple thoughts or actions
- Do NOT provide "Final Answer" until after you've executed actions and reviewed observations
- The system will execute your action and provide the REAL observation

Now, start with your Thought about what you need to find, then provide your Action.""")
        ]
        
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Get response from LLM
            response = self._call_llm(messages)
            
            # Try to parse an action first
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
                messages.append(UserMessage(content=f"""Observation: {observation}

⚠️ CRITICAL: Provide ONLY your next step - do NOT simulate the entire conversation!

Based on this observation, provide EITHER:

1. ONE more Thought + Action if you need more information:
   Thought: [what you're thinking]
   Action: {{"tool": "...", "parameters": {{...}}}}
   [STOP HERE - do not write Step 2, do not write Observation, do not write Final Answer yet]

2. OR Final Answer if you have enough information:
   Final Answer: [your complete answer based on ALL observations you've received]

Remember: 
- Provide ONLY ONE Thought and ONE Action per response
- Do NOT write "Step 1, Step 2, etc."
- Do NOT write "Observation:" (the system provides that)
- Do NOT provide Final Answer until you've actually gathered enough information"""))
            
            else:
                # No action found - check if this is a final answer
                final_answer = self._extract_final_answer(response)
                
                if final_answer:
                    # This is a legitimate final answer
                    return final_answer
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

