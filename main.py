from visualize import LearningGraphParser, MermaidGenerator, GraphvizGenerator
import argparse
from pathlib import Path


def generate_all_outputs(yaml_dir: str, output_dir: str):
    """모든 출력 파일 생성"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 파싱
    parser = LearningGraphParser(yaml_dir)
    parser.load_all()

    mermaid_gen = MermaidGenerator(parser)
    graphviz_gen = GraphvizGenerator(parser)

    # 메타 그래프 (도메인 간 관계)
    meta_mermaid = mermaid_gen.generate_meta_graph()
    with open(output_path / "meta_graph.mermaid", "w", encoding="utf-8") as f:
        f.write(meta_mermaid)
    print("✓ Generated: meta_graph.mermaid")

    # 전체 그래프
    full_mermaid = mermaid_gen.generate_full_graph()
    with open(output_path / "full_graph.mermaid", "w", encoding="utf-8") as f:
        f.write(full_mermaid)
    print("✓ Generated: full_graph.mermaid")

    full_dot = graphviz_gen.generate_full_graph()
    with open(output_path / "full_graph.dot", "w", encoding="utf-8") as f:
        f.write(full_dot)
    print("✓ Generated: full_graph.dot")

    # 도메인별 그래프
    for domain_id in parser.domains.keys():
        # Mermaid
        domain_mermaid = mermaid_gen.generate_domain_graph(domain_id)
        with open(
            output_path / f"{domain_id}_graph.mermaid", "w", encoding="utf-8"
        ) as f:
            f.write(domain_mermaid)
        print(f"✓ Generated: {domain_id}_graph.mermaid")

        # DOT
        domain_dot = graphviz_gen.generate_domain_graph(domain_id)
        with open(output_path / f"{domain_id}_graph.dot", "w", encoding="utf-8") as f:
            f.write(domain_dot)
        print(f"✓ Generated: {domain_id}_graph.dot")

    # DOT → PNG 변환 시도
    print("\n📊 Generating PNG images...")
    try:
        import subprocess

        for dot_file in output_path.glob("*.dot"):
            png_file = dot_file.with_suffix(".png")
            result = subprocess.run(
                ["dot", "-Tpng", "-Gdpi=150", str(dot_file), "-o", str(png_file)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"✓ Generated: {png_file.name}")
            else:
                print(f"✗ Failed: {png_file.name} - {result.stderr}")
    except FileNotFoundError:
        print(
            "⚠ Graphviz not installed. DOT files generated but PNG conversion skipped."
        )
        print("  Install with: apt-get install graphviz")

    print(f"\n✅ All outputs saved to: {output_path}")

    # 통계 출력
    print("\n📈 Statistics:")
    print(f"   Domains: {len(parser.domains)}")
    print(f"   Nodes: {len(parser.nodes)}")
    print(f"   Edges: {len(parser.edges)}")
    print(f"   - Internal: {sum(1 for e in parser.edges if e.edge_type == 'internal')}")
    print(f"   - External: {sum(1 for e in parser.edges if e.edge_type == 'external')}")


def main():
    arg_parser = argparse.ArgumentParser(description="AI Learning Graph Visualizer")
    arg_parser.add_argument(
        "--yaml-dir",
        default="./learning-graphs",
        help="Directory containing YAML files",
    )
    arg_parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Output directory for generated files",
    )

    args = arg_parser.parse_args()
    generate_all_outputs(args.yaml_dir, args.output_dir)


if __name__ == "__main__":
    main()
