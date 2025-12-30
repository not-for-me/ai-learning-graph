# AI Learning Graph - Agent Context

## Project Overview

This is a **knowledge graph-based learning system** designed to systematically study AI-related technologies. The core concept is to break down complex AI topics into granular concepts (nodes) and visualize their relationships (edges) in a directed graph format.

### Problem Statement

Learning all AI technologies from scratch is time-prohibitive. This project addresses this by:
1. Decomposing topics into atomic concepts with clear prerequisites
2. Enabling rapid self-assessment through quizzes at each node
3. Focusing revision efforts on forgotten or misunderstood concepts
4. Maintaining mastery tracking to optimize learning path

### Use Case

Users interact with the graph to:
- Quickly verify understanding of specific concepts
- Identify knowledge gaps through quiz failures
- Follow optimal learning paths based on prerequisite relationships
- Track mastery levels over time

## Technical Architecture

### Technology Stack

- **Language**: Python 3.14+
- **Package Manager**: `uv` (fast Python package installer)
- **Linter/Formatter**: `ruff` (replacing black, flake8, isort)
- **Type Checking**: Python type hints (potentially mypy or pyright)
- **Data Format**: YAML for graph definitions
- **Visualization**: Mermaid diagrams, Graphviz (DOT → PNG)

### Project Structure

```
ai-learning-graph/
├── learning-graphs/          # YAML graph definitions
│   ├── 00_graph_meta.yaml   # Domain relationships & global config
│   ├── 01_math_foundations.yaml
│   ├── 02_deep_learning.yaml
│   ├── 03_transformer.yaml
│   ├── 04_llm.yaml
│   └── 05_agent.yaml
├── visualize.py             # Core parser and visualization generators
├── main.py                  # CLI entry point
├── pyproject.toml           # Project config & dependencies
└── uv.lock                  # Locked dependencies
```

### Core Components

#### 1. YAML Graph Schema

Each domain YAML file contains:

```yaml
domain:
  id: domain_id
  name: "Domain Name"

nodes:
  - id: node_id
    name: "Concept Name"
    category: category_id
    depth: 0                    # Learning depth (0 = entry point)
    mastery: 0                  # 0: unlearned, 1: learning, 2: understood, 3: mastered
    prerequisites: [...]         # Internal dependencies (same domain)
    external_prerequisites:      # External dependencies (other domains)
      - {domain: domain_id, node: node_id}
    core_concepts: [...]         # Key learning points
    quiz: [...]                  # Self-assessment questions
    resources: [...]             # Learning materials

categories:
  category_id:
    name: "Category Name"
    color: "#HEX"
```

**Key Design Decisions**:
- **Node IDs**: Scoped as `domain:node_id` to avoid collisions
- **Edge Types**:
  - `internal`: Within same domain (prerequisites)
  - `external`: Cross-domain dependencies
- **Depth Levels**: Used for topological visualization (not strict layers)

#### 2. Python Parser (`visualize.py`)

**Classes**:

1. `LearningGraphParser`
   - Loads all YAML files from a directory
   - Builds unified graph: `nodes: dict[str, Node]`, `edges: list[Edge]`
   - Handles both internal and external prerequisites

2. `MermaidGenerator`
   - `generate_full_graph()`: All domains in subgraphs
   - `generate_domain_graph(domain_id)`: Single domain with external deps
   - `generate_meta_graph()`: Domain-level relationships only

3. `GraphvizGenerator`
   - Similar to Mermaid but outputs DOT format
   - Supports PNG rendering via `dot` command

#### 3. Main Script (`main.py`)

- Orchestrates parsing and generation
- Outputs to `./graph-outputs/` by default:
  - `meta_graph.mermaid` - Domain relationships
  - `full_graph.{mermaid,dot,png}` - Complete graph
  - `{domain}_graph.{mermaid,dot,png}` - Per-domain graphs
- Prints statistics (node/edge counts)

## Graph Semantics

### Domain Structure

Currently defined domains (in learning order):
1. **math_foundations**: Linear algebra, calculus, probability/statistics
2. **deep_learning**: Neural networks, backpropagation, CNN, RNN
3. **transformer**: Attention mechanism, encoder-decoder architecture
4. **llm**: Large language models, prompting, fine-tuning, RLHF
5. **agent**: Tool use, RAG, multi-agent systems, planning

**Domain Dependencies** (from `graph_meta.yaml`):
- `math_foundations → deep_learning` (strong)
- `deep_learning → transformer` (strong)
- `transformer → llm` (strong)
- `llm → agent` (strong)

### Node Relationships

**Prerequisites** establish a DAG (Directed Acyclic Graph):
- Edges point from prerequisite → dependent concept
- Example: `vector → dot_product → cosine_similarity → attention_score`

**Cross-Domain Links**:
- Connect foundational math to advanced concepts
- Example: `math_foundations:gradient → deep_learning:gradient_descent`

### Mastery Tracking

Nodes have `mastery` field (0-3):
- **0**: Unlearned
- **1**: Currently learning
- **2**: Understood (passed quiz)
- **3**: Mastered (can teach others)

`graph_meta.yaml` defines:
- `mastery_threshold: 0.8` (80% quiz accuracy)
- `review_interval_days: 14` (spaced repetition)

## Development Guidelines

### When Adding New Concepts

1. **Identify the domain**: Choose existing or propose new domain
2. **Define prerequisites**: List both internal and external dependencies
3. **Set appropriate depth**: Based on longest path from entry nodes
4. **Write 2-3 quiz questions**:
   - `calculation`: Numerical/symbolic computation
   - `conceptual`: Explain the "why"
   - `application`: Real-world AI use case
5. **Add resources**: Videos, book chapters, papers

### When Modifying Graph Structure

- Ensure no cycles (DAG property)
- Update `cross_domain_connections` in domain YAML
- Run `python main.py` to regenerate visualizations
- Verify output graphs visually for correctness

### Code Contributions

- Use `uv` for dependency management: `uv pip install <package>`
- Run `ruff check .` before committing
- Add type hints to new functions
- Follow existing naming conventions:
  - `snake_case` for functions/variables
  - `PascalCase` for classes

### Visualization Best Practices

- **Colors**: Use distinct colors per category (defined in YAML)
- **External Nodes**: Show with dashed borders in domain graphs
- **Meta Graph**: Keep simple (domain-level only) for high-level overview

## Common Tasks

### Generate All Outputs
```bash
python main.py --yaml-dir ./learning-graphs --output-dir ./graph-outputs
```

### Add New Domain
1. Create `06_new_domain.yaml` in `learning-graphs/`
2. Update `graph_meta.yaml` domains list
3. Add domain dependencies
4. Regenerate graphs

### Update Mastery Levels
- Manually edit YAML `mastery:` field
- Future: Build interactive CLI/web interface for quiz-taking

### Export for Different Tools
- **Mermaid**: Paste `.mermaid` files into GitHub/Notion
- **Graphviz**: Use `.dot` files or `.png` images
- **Future**: Neo4j, Obsidian graph view

## Known Limitations

1. **Manual Mastery Updates**: No automated quiz system yet
2. **No Progress Tracking**: `graph_meta.yaml` progress fields unused
3. **Static Visualization**: No interactive graph exploration
4. **No Spaced Repetition Logic**: Settings defined but not implemented

## Future Enhancements

- [ ] Interactive CLI for taking quizzes
- [ ] Spaced repetition scheduler
- [ ] Web UI for graph exploration
- [ ] Export to Anki flashcards
- [ ] Git-based progress tracking
- [ ] Recommendation system for next concepts to study

## Important Context for Agents

When working on this project:
1. **Preserve DAG structure**: Never introduce cycles
2. **Maintain YAML schema**: All nodes must have required fields
3. **Unique node IDs**: Always namespace with `domain:`
4. **Quiz quality**: Questions should test deep understanding, not rote memorization
5. **Realistic resources**: Only add resources you can verify exist

This is a **personal learning tool**, so optimize for:
- Fast self-assessment
- Clear prerequisite chains
- Minimal friction in tracking progress
- Flexible enough to adapt as learning progresses
