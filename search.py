import os
from typing import Any, Dict, Iterable, Optional

from dotenv import load_dotenv
import requests

load_dotenv()


class GitHubSearchClient:
	"""
	Simple wrapper for GitHub Search API (v2022-11-28).
	"""

	def __init__(
		self,
		token: Optional[str] = None,
		base_url: str = "https://api.github.com",
		api_version: str = "2022-11-28",
		timeout: int = 15,
	):
		self.base_url = base_url.rstrip("/")
		self.timeout = timeout
		self.session = requests.Session()

		if token is None:
			token = os.getenv("GITHUB_TOKEN")

		headers = {
			"Accept": "application/vnd.github+json",
			"X-GitHub-Api-Version": api_version,
		}
		if token:
			headers["Authorization"] = f"Bearer {token}"
		self.session.headers.update(headers)

	def _build_query(self, q: Optional[str], qualifiers: Optional[Dict[str, Any]]) -> str:
		parts = []
		if q:
			q = q.strip()
			if q:
				parts.append(q)
		if qualifiers:
			for key, value in qualifiers.items():
				if value is None or value == "":
					continue
				if isinstance(value, (list, tuple, set)):
					for v in value:
						parts.append(f"{key}:{v}")
				else:
					parts.append(f"{key}:{value}")
		return " ".join(str(p).strip() for p in parts if str(p).strip())

	def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		import time

		url = f"{self.base_url}{path}"
		clean_params = {k: v for k, v in (params or {}).items() if v is not None}
		resp = self.session.request(method, url, params=clean_params, timeout=self.timeout)

		# Rate limit handling
		if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
			reset = resp.headers.get("X-RateLimit-Reset")
			wait = None
			if reset:
				try:
					wait = max(0, int(reset) - int(time.time()))
				except Exception:
					wait = None
			msg = "GitHub API rate limit exceeded."
			if wait is not None:
				msg += f" Retry after ~{wait}s."
			raise RuntimeError(msg)

		if not resp.ok:
			try:
				data = resp.json()
				err = data.get("message", resp.text)
			except Exception:
				err = resp.text
			raise RuntimeError(f"GitHub API error {resp.status_code}: {err}")

		return resp.json()

	def search_code(
		self,
		q: str,
		qualifiers: Optional[Dict[str, Any]] = None,
		sort: Optional[str] = None,  # 'indexed'
		order: Optional[str] = None,  # 'desc' or 'asc'
		per_page: int = 30,
		page: int = 1,
	) -> Dict[str, Any]:
		"""
		GET /search/code
		Example qualifiers: {'repo': 'owner/name', 'org': 'myorg', 'in': 'file', 'language': 'python', 'filename': 'dockerfile'}
		"""
		query = self._build_query(q, qualifiers)
		return self._request(
			"GET",
			"/search/code",
			params={
				"q": query,
				"sort": sort,
				"order": order,
				"per_page": per_page,
				"page": page,
			},
		)

	def search_issues(
		self,
		q: str,
		qualifiers: Optional[Dict[str, Any]] = None,
		sort: Optional[str] = None,  # 'comments' | 'created' | 'updated'
		order: Optional[str] = None,  # 'desc' or 'asc'
		per_page: int = 30,
		page: int = 1,
	) -> Dict[str, Any]:
		"""
		GET /search/issues
		Example qualifiers: {'repo': 'owner/name', 'is': 'issue', 'state': 'open', 'label': 'bug', 'assignee': 'octocat'}
		"""
		query = self._build_query(q, qualifiers)
		return self._request(
			"GET",
			"/search/issues",
			params={
				"q": query,
				"sort": sort,
				"order": order,
				"per_page": per_page,
				"page": page,
			},
		)

	def paginate_items(
		self,
		endpoint: str,  # '/search/code' or '/search/issues'
		q: str,
		qualifiers: Optional[Dict[str, Any]] = None,
		sort: Optional[str] = None,
		order: Optional[str] = None,
		per_page: int = 100,
		max_pages: Optional[int] = None,
	) -> Iterable[Dict[str, Any]]:
		"""
		Yield items across pages until exhausted or max_pages reached.
		"""
		page = 1
		while True:
			data = self._request(
				"GET",
				endpoint,
				params={
					"q": self._build_query(q, qualifiers),
					"sort": sort,
					"order": order,
					"per_page": per_page,
					"page": page,
				},
			)
			items = data.get("items", []) or []
			for item in items:
				yield item
			if len(items) < per_page:
				break
			page += 1
			if max_pages is not None and page > max_pages:
				break

	def get_repository_tree(
		self,
		owner: str,
		repo: str,
		branch: Optional[str] = None,
		recursive: bool = True,
	) -> Dict[str, Any]:
		"""
		GET /repos/{owner}/{repo}/git/trees/{tree_sha}
		Fetches the entire repository structure (directories and files).
		If branch is not specified, uses the default branch.
		"""
		# First, get the default branch if not specified
		if branch is None:
			repo_data = self._request("GET", f"/repos/{owner}/{repo}")
			branch = repo_data.get("default_branch", "main")
		
		# Get the branch to get the commit SHA
		branch_data = self._request("GET", f"/repos/{owner}/{repo}/branches/{branch}")
		tree_sha = branch_data["commit"]["commit"]["tree"]["sha"]
		
		# Get the tree (recursive to get all files)
		params = {"recursive": "1"} if recursive else {}
		tree_data = self._request("GET", f"/repos/{owner}/{repo}/git/trees/{tree_sha}", params=params)
		
		return tree_data

	def get_file_contents(
		self,
		owner: str,
		repo: str,
		path: str,
		branch: Optional[str] = None,
	) -> Dict[str, Any]:
		"""
		GET /repos/{owner}/{repo}/contents/{path}
		Fetches the contents of a file from a GitHub repository.
		Returns the file metadata and content (base64 encoded for binary files).
		"""
		params = {}
		if branch:
			params["ref"] = branch
		
		return self._request("GET", f"/repos/{owner}/{repo}/contents/{path}", params=params)
