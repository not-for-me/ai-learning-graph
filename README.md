# AI Learning Graph

A knowledge graph-based system for systematic self-study of AI technologies, from mathematical foundations to agent systems.

## Why This Project?

Learning AI from scratch is overwhelming. This project helps you:

- **Focus on gaps**: Quickly test your understanding and identify weak spots
- **Follow the right path**: Prerequisites are explicit, no guessing what to learn next
- **Track progress**: Visual mastery tracking across 100+ concepts
- **Optimize time**: Study what you don't know, skip what you've mastered

## How It Works

1. **Concepts as nodes**: Each node is an atomic concept (e.g., "Gradient Descent", "Attention Mechanism")
2. **Prerequisites as edges**: Arrows show dependencies (learn A before B)
3. **Self-assessment quizzes**: Each node has 2-3 questions to verify understanding
4. **Mastery levels**: Track your progress from 0 (unlearned) to 3 (mastered)

## Project Structure

```
ai-learning-graph/
├── learning-graphs/          # YAML files defining concepts and relationships
│   ├── 00_graph_meta.yaml   # Domain-level structure
│   ├── 01_math_foundations.yaml
│   ├── 02_deep_learning.yaml
│   ├── 03_transformer.yaml
│   ├── 04_llm.yaml
│   └── 05_agent.yaml
├── visualize.py             # Graph parser and visualization generators
├── main.py                  # CLI tool to generate outputs
└── graph-outputs/           # Generated diagrams (Mermaid, DOT, PNG)
```

## Learning Domains

### 1. Math Foundations
- Linear Algebra: Vectors, matrices, eigenvalues
- Calculus: Derivatives, chain rule, gradients
- Probability: Conditional probability, Bayes theorem, MLE

### 2. Deep Learning
- Neural networks, backpropagation
- CNNs, RNNs, optimization techniques

### 3. Transformer Architecture
- Self-attention mechanism
- Encoder-decoder structure
- Positional encoding

### 4. Large Language Models
- Pretraining, fine-tuning, RLHF
- Prompting strategies
- Tokenization, embeddings

### 5. AI Agents
- Tool use, function calling
- RAG (Retrieval-Augmented Generation)
- Multi-agent systems, planning

## Quick Start

### Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager
- (Optional) Graphviz for PNG generation

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-learning-graph

# Install dependencies
uv sync

# (Optional) Install Graphviz for image output
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz
```

### Generate Visualizations

```bash
# Generate all outputs (Mermaid, DOT, PNG)
python main.py

# Custom directories
python main.py --yaml-dir ./learning-graphs --output-dir ./outputs
```

### Output Files

After running, check `graph-outputs/`:

- `meta_graph.mermaid` - High-level domain relationships
- `full_graph.{mermaid,dot,png}` - Complete concept graph
- `{domain}_graph.{mermaid,dot,png}` - Individual domain graphs

## Using the Graphs

### View Mermaid Diagrams
- Paste `.mermaid` files into [Mermaid Live Editor](https://mermaid.live/)
- Or render directly in GitHub markdown:

````markdown
```mermaid
<paste content here>
```
````

### View PNG Images
Open `.png` files directly or include in notes/documents.

## Example Node Structure

Each concept in the YAML files looks like:

```yaml
- id: gradient_descent
  name: "Gradient Descent"
  category: optimization
  depth: 2
  mastery: 0  # Update as you learn

  prerequisites:
    - gradient
    - learning_rate

  core_concepts:
    - "Iterative optimization by following negative gradient"
    - "Learning rate controls step size"
    - "Variants: SGD, mini-batch, momentum"

  quiz:
    - question: "Why move in the negative gradient direction?"
      answer: "Gradient points to steepest increase; negative direction minimizes loss"
      type: conceptual

  resources:
    - type: video
      title: "StatQuest: Gradient Descent"
      url: "https://www.youtube.com/watch?v=..."
```

## Tracking Your Progress

### Manual Method (Current)

1. Study a concept and complete the quiz
2. Edit the YAML file:
   ```yaml
   mastery: 2  # 0→1→2→3
   last_reviewed: "2025-12-30"
   ```
3. Regenerate graphs to see updated progress

### Future: Interactive CLI (Planned)

```bash
python study.py --domain math_foundations
# Interactive quiz, automatic mastery updates
```

## Customization

### Add New Concepts

Edit the appropriate domain YAML:

```yaml
nodes:
  - id: your_new_concept
    name: "Your Concept Name"
    category: existing_category
    depth: 2
    prerequisites:
      - prerequisite_node_id
    core_concepts:
      - "Key point 1"
      - "Key point 2"
    quiz:
      - question: "Test question?"
        answer: "Expected answer"
        type: conceptual
```

### Add New Domain

1. Create `06_your_domain.yaml` in `learning-graphs/`
2. Update `00_graph_meta.yaml`:
   ```yaml
   domains:
     - id: your_domain
       name: "Your Domain Name"
       file: "06_your_domain.yaml"
   ```
3. Add domain dependencies
4. Regenerate graphs

## Development

### Tech Stack

- **Language**: Python 3.14+
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Linter**: [ruff](https://github.com/astral-sh/ruff)
- **Type Hints**: Python native typing

### Setup Development Environment

```bash
# Install dev dependencies
uv sync --all-groups

# Run linter
ruff check .

# Format code
ruff format .
```

### Project Philosophy

- **Simplicity over features**: Focus on core learning workflow
- **YAML as source of truth**: No database, just version-controlled files
- **Visualization first**: Seeing the graph is crucial for understanding
- **Self-paced**: No forced scheduling, learn at your own rhythm

## Roadmap

- [x] YAML graph definition
- [x] Mermaid & Graphviz generation
- [x] Math foundations domain
- [x] Deep learning, Transformer, LLM, Agent domains
- [ ] Interactive quiz CLI
- [ ] Spaced repetition scheduler
- [ ] Web UI for graph exploration
- [ ] Export to Anki/Obsidian
- [ ] Progress analytics & recommendations

## Contributing

This is a personal learning tool, but suggestions are welcome:

1. **Add concepts**: Submit PRs with new nodes (include quizzes!)
2. **Improve quizzes**: Better questions that test understanding
3. **Fix errors**: Corrections to concepts or relationships
4. **Add resources**: High-quality learning materials

## License

MIT License - Feel free to fork and adapt for your own learning journey.

## Acknowledgments

Inspired by:
- [Curriculum for learning ML](https://github.com/microsoft/ML-For-Beginners)
- Knowledge graph approaches to learning
- Spaced repetition systems (Anki, SuperMemo)

---

**Happy Learning!** Start with `01_math_foundations.yaml` and work your way up to building AI agents.
