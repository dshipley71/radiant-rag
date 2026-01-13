# How AGENTS.md is Used in Radiant RAG

## File Structure

```
radiant-rag/
├── AGENTS.md                    # Root: Project overview, setup, common tasks
└── radiant/
    └── agents/
        └── AGENTS.md            # Nested: Agent-specific development guide
```

## Why Two Files?

### Root AGENTS.md
- **Audience**: Anyone working on any part of the project
- **Content**: Setup commands, architecture overview, testing, code style
- **Use case**: "How do I run tests?" "What's the project structure?"

### Nested radiant/agents/AGENTS.md  
- **Audience**: Developers modifying or creating agents
- **Content**: Agent hierarchy, parameter signatures, implementation patterns
- **Use case**: "How do I create a new agent?" "What parameters does CriticAgent expect?"

## How AI Coding Agents Use These Files

### 1. Automatic Discovery
Most AI coding tools (Cursor, Copilot, Claude, Windsurf, etc.) automatically:
- Scan for AGENTS.md at project root
- Find nested AGENTS.md files in subdirectories
- Use the **nearest** AGENTS.md to the file being edited

### 2. Context Loading
When you ask an AI assistant to modify code:
```
User: "Add a new caching agent to the pipeline"

AI Assistant internally:
1. Finds AGENTS.md in project root → learns setup/testing commands
2. Finds radiant/agents/AGENTS.md → learns agent patterns
3. Uses both to generate correct code with proper signatures
```

### 3. Proximity Rules
If editing `radiant/agents/planning.py`:
- Primary context: `radiant/agents/AGENTS.md` (closest)
- Secondary context: Root `AGENTS.md`

If editing `radiant/orchestrator.py`:
- Primary context: Root `AGENTS.md`

## How Radiant RAG Could Index AGENTS.md

### Option 1: Priority Document Source
```python
# In ingestion pipeline
class DocumentPrioritizer:
    HIGH_PRIORITY_PATTERNS = ["AGENTS.md", "README.md", "CONTRIBUTING.md"]
    
    def get_priority(self, filename: str) -> float:
        if filename in self.HIGH_PRIORITY_PATTERNS:
            return 1.0  # Always include in retrieval
        return 0.5
```

### Option 2: Automatic Context Injection
```python
# In PlanningAgent
def _before_execute(self, query: str, **kwargs):
    if self._is_code_related(query):
        agents_md = self._load_nearest_agents_md(kwargs.get("file_context"))
        if agents_md:
            self._inject_context(agents_md)
```

### Option 3: Specialized Retriever
```python
class AgentsMdRetriever:
    """Always retrieve AGENTS.md for development queries."""
    
    def retrieve(self, query: str, project_path: str) -> List[str]:
        if self._is_dev_query(query):
            return self._find_all_agents_md(project_path)
        return []
```

## Benefits of This Structure

| Benefit | How It Helps |
|---------|--------------|
| **Separation of concerns** | General info vs agent-specific details |
| **Proximity-based relevance** | AI gets most relevant context first |
| **Maintainability** | Update agent docs without touching root |
| **Scalability** | Add more nested files as project grows |
| **Standard format** | Works with 20+ AI coding tools |

## Example: AI Assistant Workflow

```
User: "Fix the parameter mismatch error in RRFAgent call"

AI reads root AGENTS.md:
- Learns: "Parameter names must match _execute() signature"
- Learns: Testing command is `pytest tests/ -v`

AI reads radiant/agents/AGENTS.md:
- Finds: RRFAgent expects `runs: List[List[Tuple]]`
- Finds: Common mistake section showing exact fix

AI generates fix:
- Changes `retrieval_lists=` to `runs=`
- Runs tests to verify
```

## Extending the Structure

For larger projects, you could add more nested files:

```
radiant-rag/
├── AGENTS.md                      # Project root
├── radiant/
│   ├── agents/
│   │   └── AGENTS.md              # Agent development
│   ├── storage/
│   │   └── AGENTS.md              # Vector store implementations
│   ├── ingestion/
│   │   └── AGENTS.md              # Document processing
│   └── orchestrator/
│       └── AGENTS.md              # Pipeline coordination
└── tests/
    └── AGENTS.md                  # Testing conventions
```

Each nested file provides context specific to that subsystem.
