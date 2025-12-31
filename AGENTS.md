# AI Learning Graph - Agent Context

## Project Overview

This is a **knowledge graph-based learning system** designed to systematically study AI-related technologies. The core concept is to break down complex AI topics into granular concepts (nodes) and visualize their relationships (edges) in a directed graph format.

### Problem Statement

Learning all AI technologies from scratch is time-prohibitive. This project addresses this by:
1. Decomposing topics into atomic concepts with clear prerequisites
2. Enabling rapid self-assessment through quizzes at each node
3. Focusing revision efforts on forgotten or misunderstood concepts
4. Maintaining mastery tracking to optimize learning path
5. **Capturing deep-dive questions from actual learning sessions for review**

### Use Case

Users interact with the graph to:
- Quickly verify understanding of specific concepts
- Identify knowledge gaps through quiz failures
- Follow optimal learning paths based on prerequisite relationships
- Track mastery levels over time
- **Document clarifying questions that emerge during study for spaced repetition**

## Technical Architecture

### Technology Stack

- **Language**: Python 3.14+
- **Package Manager**: `uv` (fast Python package installer)
- **Linter/Formatter**: `ruff` (replacing black, flake8, isort)
- **Type Checking**: Python type hints (potentially mypy or pyright)
- **Data Format**: YAML for graph definitions
- **Visualization**: Mermaid diagrams, Graphviz (DOT → PNG)
- **Review System**: Markdown-based FAQ documents

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
├── reviews/                  # Learning session review questions
│   ├── 01_math_foundations_faq.md
│   ├── 02_deep_learning_faq.md
│   ├── 03_transformer_faq.md
│   ├── 04_llm_faq.md
│   ├── 05_agent_faq.md
│   └── README.md            # Review system guide
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

#### 4. Review System (`reviews/`)

**Purpose**: Capture deep-dive questions and insights from actual learning sessions that go beyond the predefined quizzes in YAML files.

**Structure**:
- One FAQ file per domain (matches YAML domain structure)
- Date-stamped sections for chronological tracking
- Links to source conversations/materials
- Cross-references to related nodes in the graph
- Confidence levels for spaced repetition

**Workflow**:
1. During study, when encountering unclear concepts, document questions
2. After clarification (via Claude, papers, videos), record answers in domain FAQ
3. Include metadata: date, related node IDs, confidence level, source links
4. Review periodically based on confidence levels

**FAQ Document Template**:
```markdown
# [Domain Name] - Review Questions

> Questions and insights from actual learning sessions

## YYYY-MM-DD: [Topic/Session Name]

### Q: [Question that emerged during study]

**Context**: [What you were studying when this question arose]
**Source**: [Link to conversation, paper, video, etc.]

**Answer**: 
- [Key insight 1]
- [Key insight 2]
- [Key insight 3]

**Related nodes**: 
- `domain:node_id` - [Brief connection explanation]
- `domain:node_id` - [Brief connection explanation]

**Confidence**: ⭐⭐⭐ (3/3 - can teach others) | ⭐⭐ (2/3 - understood) | ⭐ (1/3 - learning)

**Review history**:
- YYYY-MM-DD: First learned
- YYYY-MM-DD: Reviewed, confidence ⭐ → ⭐⭐

---
```

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

### When Adding Review Questions

1. **Identify the domain**: Match with existing domain files
2. **Create date-stamped section**: Use ISO format (YYYY-MM-DD)
3. **Write clear question**: What specifically was unclear?
4. **Provide context**: What were you studying? Why did this question arise?
5. **Document answer thoroughly**: Include multiple perspectives if relevant
6. **Link to nodes**: Reference related concepts in graph using `domain:node_id` format
7. **Set confidence level**: Honest self-assessment (1-3 stars)
8. **Add source**: Link to conversations, papers, videos for future reference

### When to Use Reviews vs YAML Quizzes

**YAML Quizzes (`quiz:` field)**:
- Standard verification questions for each concept
- Test core understanding required to proceed
- Designed for repeated use (spaced repetition)
- Should be relatively stable

**Review FAQs (`reviews/` folder)**:
- Specific questions that arose during your learning
- Edge cases, nuances, or connections between concepts
- Personal learning journey documentation
- May evolve as understanding deepens

### Review Maintenance

- **Weekly**: Add new questions from study sessions
- **Monthly**: Review low-confidence (⭐) questions, update confidence
- **Quarterly**: Archive fully mastered (⭐⭐⭐) questions to separate file if desired

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

### Document Learning Session Questions
```bash
# Example workflow
cd reviews/

# While studying transformers and encounter confusion
echo "## $(date +%Y-%m-%d): Attention Mechanism Deep Dive" >> 03_transformer_faq.md
echo "" >> 03_transformer_faq.md
echo "### Q: Why is softmax applied to attention scores?" >> 03_transformer_faq.md
# ... then manually fill in context, answer, related nodes, etc.
```

### Review Past Questions
```bash
# Search for specific topic across all FAQs
grep -r "gradient" reviews/*.md

# Find all low-confidence questions
grep -r "⭐ (" reviews/*.md
```

## Known Limitations

1. **Manual Mastery Updates**: No automated quiz system yet
2. **No Progress Tracking**: `graph_meta.yaml` progress fields unused
3. **Static Visualization**: No interactive graph exploration
4. **No Spaced Repetition Logic**: Settings defined but not implemented
5. **Manual Review Documentation**: No automated extraction from study sessions

## Future Enhancements

- [ ] Interactive CLI for taking quizzes
- [ ] Spaced repetition scheduler
- [ ] Web UI for graph exploration
- [ ] Export to Anki flashcards
- [ ] Git-based progress tracking
- [ ] Recommendation system for next concepts to study
- [ ] **Automated review question extraction from study sessions**
- [ ] **CLI tool to add review questions interactively**
- [ ] **Confidence-based review scheduler**

## Important Context for Agents

When working on this project:
1. **Preserve DAG structure**: Never introduce cycles
2. **Maintain YAML schema**: All nodes must have required fields
3. **Unique node IDs**: Always namespace with `domain:`
4. **Quiz quality**: Questions should test deep understanding, not rote memorization
5. **Realistic resources**: Only add resources you can verify exist
6. **Review questions**: Should capture real learning moments, not manufactured examples
7. **Link reviews to graph**: Always reference related node IDs for traceability

This is a **personal learning tool**, so optimize for:
- Fast self-assessment
- Clear prerequisite chains
- Minimal friction in tracking progress
- Flexible enough to adapt as learning progresses
- **Capturing authentic learning experiences for review**