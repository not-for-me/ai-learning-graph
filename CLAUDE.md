# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a knowledge graph-based system for systematic self-study of AI technologies. The project uses YAML files to define learning concepts and their relationships, then generates visual graphs (Mermaid, Graphviz DOT, PNG) to help track learning progress from mathematical foundations to AI agent systems.

## Development Commands

### Setup
```bash
# Install dependencies
uv sync

# Install dev dependencies (includes ruff linter)
uv sync --all-groups
```

### Generate Visualizations
```bash
# Generate all outputs (Mermaid, DOT, PNG files)
python main.py

# Custom input/output directories
python main.py --yaml-dir ./learning-graphs --output-dir ./outputs
```

### Code Quality
```bash
# Run linter
ruff check .

# Format code
ruff format .
```

### Optional: PNG Generation
Requires Graphviz to be installed separately:
```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz
```

## Architecture

### Core Components

**1. YAML Data Structure**
- `learning-graphs/00_graph_meta.yaml`: Defines domain-level structure and learning paths
- `learning-graphs/01-05_*.yaml`: Individual domain files (math, deep learning, transformer, LLM, agent)
- Each YAML contains nodes (concepts) with prerequisites, quizzes, and resources

**2. Graph Processing Pipeline** (`visualize.py`)
- `LearningGraphParser`: Loads YAML files and builds internal graph structure
  - Nodes: Individual concepts with id, name, category, depth, mastery level
  - Edges: Two types - "internal" (within domain) and "external" (cross-domain)
  - Namespacing: Node IDs are prefixed with domain (e.g., `math_foundations:gradient`)
- `MermaidGenerator`: Creates Mermaid flowchart syntax
  - `generate_meta_graph()`: High-level domain relationships
  - `generate_full_graph()`: All domains with subgraphs
  - `generate_domain_graph(domain_id)`: Single domain with external dependencies shown
- `GraphvizGenerator`: Creates DOT format for PNG rendering
  - Similar structure to Mermaid but with cluster subgraphs
  - External dependencies shown with dashed borders

**3. CLI Tool** (`main.py`)
- `generate_all_outputs()`: Orchestrates full pipeline
- Generates 3 types of outputs for each domain:
  - `.mermaid` files (paste into mermaid.live or GitHub)
  - `.dot` files (Graphviz source)
  - `.png` images (if Graphviz available)

### Node Structure in YAML

```yaml
- id: gradient_descent          # Used for internal references
  name: "Gradient Descent"      # Display name
  category: optimization        # Groups nodes visually
  depth: 2                      # For vertical layout
  mastery: 0                    # 0-3 tracking level

  prerequisites:                # Same-domain dependencies
    - gradient
    - learning_rate

  external_prerequisites:       # Cross-domain dependencies
    - domain: math_foundations
      node: derivative

  core_concepts:                # Learning points
    - "Iterative optimization..."

  quiz:                         # Self-assessment
    - question: "Why negative gradient?"
      answer: "Points to steepest decrease"
      type: conceptual
```

### Graph Data Model

- **Node namespacing**: All node IDs are `{domain}:{local_id}` internally
- **Edge types**:
  - `internal`: Within same domain (solid arrows)
  - `external`: Cross-domain (dashed arrows)
- **Categories**: Each domain defines color-coded categories for visual grouping
- **Mastery levels**: 0=unlearned, 1=learning, 2=understood, 3=mastered

### Review System

Separate from YAML quizzes, the `reviews/` folder contains FAQ markdown files:
- Document deep-dive questions from actual learning sessions
- Include confidence levels (⭐/⭐⭐/⭐⭐⭐)
- Link back to original sources
- See `reviews/README.md` for detailed workflow

## Key Design Decisions

1. **YAML as source of truth**: No database, version-controlled files only
2. **Two-layer learning system**:
   - YAML quizzes: Breadth (verify minimum understanding)
   - Markdown FAQs: Depth (capture nuances and insights)
3. **Domain isolation**: Each domain file is self-contained except for explicit external_prerequisites
4. **Visualization-first**: Graph rendering is primary interface for understanding structure
5. **No test suite**: Project focuses on data definition and visualization generation, not complex logic
