#!/usr/bin/env python3
"""
AI Learning Graph Visualizer
YAML 기반 학습 그래프를 이미지와 Mermaid 형식으로 변환
"""

import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Node:
    id: str
    name: str
    category: str
    depth: int
    domain: str
    mastery: int = 0


@dataclass
class Edge:
    source: str
    target: str
    edge_type: str  # 'internal', 'external'
    source_domain: str
    target_domain: str

GRAPH_META_FILE = "00_graph_meta.yaml"

class LearningGraphParser:
    """YAML 파일들을 파싱하여 그래프 데이터 구조로 변환"""

    def __init__(self, yaml_dir: str):
        self.yaml_dir = Path(yaml_dir)
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.domains: dict[str, dict] = {}
        self.categories: dict[str, dict] = {}

    def load_all(self):
        """모든 YAML 파일 로드"""
        # meta.yaml 로드
        meta_path = self.yaml_dir / GRAPH_META_FILE
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = yaml.safe_load(f)

        # 도메인 파일들 로드
        for yaml_file in sorted(self.yaml_dir.glob("*.yaml")):
            if yaml_file.name == GRAPH_META_FILE:
                continue
            self._load_domain_file(yaml_file)

    def _load_domain_file(self, filepath: Path):
        """개별 도메인 YAML 파일 로드"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        domain_id = data["domain"]["id"]
        domain_name = data["domain"]["name"]
        self.domains[domain_id] = {"name": domain_name, "file": filepath.name}

        # 카테고리 저장
        if "categories" in data:
            for cat_id, cat_info in data["categories"].items():
                self.categories[f"{domain_id}:{cat_id}"] = {
                    "name": cat_info["name"],
                    "color": cat_info.get("color", "#888888"),
                    "domain": domain_id,
                }

        # 노드 파싱
        for node_data in data.get("nodes", []):
            node_id = f"{domain_id}:{node_data['id']}"
            self.nodes[node_id] = Node(
                id=node_id,
                name=node_data["name"],
                category=f"{domain_id}:{node_data['category']}",
                depth=node_data.get("depth", 0),
                domain=domain_id,
                mastery=node_data.get("mastery", 0),
            )

            # 내부 엣지 (prerequisites → 현재 노드)
            for prereq in node_data.get("prerequisites", []):
                prereq_id = f"{domain_id}:{prereq}"
                self.edges.append(
                    Edge(
                        source=prereq_id,
                        target=node_id,
                        edge_type="internal",
                        source_domain=domain_id,
                        target_domain=domain_id,
                    )
                )

            # 외부 엣지 (다른 도메인에서)
            for ext_prereq in node_data.get("external_prerequisites", []):
                ext_domain = ext_prereq["domain"]
                ext_node = ext_prereq["node"]
                ext_id = f"{ext_domain}:{ext_node}"
                self.edges.append(
                    Edge(
                        source=ext_id,
                        target=node_id,
                        edge_type="external",
                        source_domain=ext_domain,
                        target_domain=domain_id,
                    )
                )


class MermaidGenerator:
    """Mermaid 다이어그램 생성기"""

    def __init__(self, parser: LearningGraphParser):
        self.parser = parser

    def generate_full_graph(self) -> str:
        """전체 그래프 Mermaid 코드 생성"""
        lines = ["flowchart TD"]

        # 스타일 정의
        lines.append("")
        lines.append("    %% Style definitions")
        for cat_id, cat_info in self.parser.categories.items():
            style_name = cat_id.replace(":", "_")
            color = cat_info["color"]
            lines.append(
                f"    classDef {style_name} fill:{color},stroke:#333,stroke-width:1px,color:#fff"
            )

        # 도메인별 서브그래프
        lines.append("")
        for domain_id, domain_info in self.parser.domains.items():
            lines.append(f"    subgraph {domain_id}[{domain_info['name']}]")

            # 해당 도메인의 노드들
            domain_nodes = [
                n for n in self.parser.nodes.values() if n.domain == domain_id
            ]
            for node in sorted(domain_nodes, key=lambda x: x.depth):
                safe_id = node.id.replace(":", "_")
                lines.append(f'        {safe_id}["{node.name}"]')

            lines.append("    end")
            lines.append("")

        # 엣지
        lines.append("    %% Edges")
        for edge in self.parser.edges:
            source_safe = edge.source.replace(":", "_")
            target_safe = edge.target.replace(":", "_")
            if edge.edge_type == "external":
                lines.append(f"    {source_safe} -.-> {target_safe}")
            else:
                lines.append(f"    {source_safe} --> {target_safe}")

        # 스타일 적용
        lines.append("")
        lines.append("    %% Apply styles")
        for node in self.parser.nodes.values():
            safe_id = node.id.replace(":", "_")
            style_name = node.category.replace(":", "_")
            lines.append(f"    class {safe_id} {style_name}")

        return "\n".join(lines)

    def generate_domain_graph(self, domain_id: str) -> str:
        """특정 도메인만의 그래프 생성"""
        lines = ["flowchart TD"]

        # 해당 도메인의 카테고리 스타일
        lines.append("")
        lines.append("    %% Style definitions")
        for cat_id, cat_info in self.parser.categories.items():
            if cat_info["domain"] == domain_id:
                style_name = cat_id.replace(":", "_")
                color = cat_info["color"]
                lines.append(
                    f"    classDef {style_name} fill:{color},stroke:#333,stroke-width:1px,color:#fff"
                )

        # 외부 의존성 스타일
        lines.append(
            "    classDef external fill:#gray,stroke:#333,stroke-width:1px,color:#fff,stroke-dasharray: 5 5"
        )

        # depth별로 그룹화
        domain_nodes = [n for n in self.parser.nodes.values() if n.domain == domain_id]
        max_depth = max(n.depth for n in domain_nodes) if domain_nodes else 0
        print(f"max_depth: {max_depth}")

        lines.append("")

        # 노드 정의
        for node in sorted(domain_nodes, key=lambda x: (x.depth, x.category)):
            safe_id = node.id.replace(":", "_")
            lines.append(f'    {safe_id}["{node.name}"]')

        # 외부 의존성 노드 (다른 도메인에서 오는)
        external_nodes = set()
        for edge in self.parser.edges:
            if edge.target_domain == domain_id and edge.source_domain != domain_id:
                external_nodes.add(edge.source)

        if external_nodes:
            lines.append("")
            lines.append("    %% External dependencies")
            for ext_id in external_nodes:
                safe_id = ext_id.replace(":", "_")
                if ext_id in self.parser.nodes:
                    name = self.parser.nodes[ext_id].name
                    domain_name = self.parser.domains.get(ext_id.split(":")[0], {}).get(
                        "name", ""
                    )
                    lines.append(
                        f'    {safe_id}["{name}<br/><small>({domain_name})</small>"]'
                    )

        # 엣지
        lines.append("")
        lines.append("    %% Edges")
        for edge in self.parser.edges:
            if edge.target_domain == domain_id:
                source_safe = edge.source.replace(":", "_")
                target_safe = edge.target.replace(":", "_")
                if edge.edge_type == "external":
                    lines.append(f"    {source_safe} -.-> {target_safe}")
                else:
                    lines.append(f"    {source_safe} --> {target_safe}")

        # 스타일 적용
        lines.append("")
        lines.append("    %% Apply styles")
        for node in domain_nodes:
            safe_id = node.id.replace(":", "_")
            style_name = node.category.replace(":", "_")
            lines.append(f"    class {safe_id} {style_name}")

        for ext_id in external_nodes:
            safe_id = ext_id.replace(":", "_")
            lines.append(f"    class {safe_id} external")

        return "\n".join(lines)

    def generate_meta_graph(self) -> str:
        """도메인 간 관계만 보여주는 메타 그래프"""
        lines = ["flowchart LR"]

        lines.append("")
        lines.append("    %% Domain nodes")
        for domain_id, domain_info in self.parser.domains.items():
            lines.append(f'    {domain_id}[("{domain_info["name"]}")]')

        # 도메인 간 연결 (외부 엣지에서 추출)
        domain_edges = set()
        for edge in self.parser.edges:
            if edge.edge_type == "external":
                domain_edges.add((edge.source_domain, edge.target_domain))

        lines.append("")
        lines.append("    %% Domain relationships")
        for source_domain, target_domain in sorted(domain_edges):
            lines.append(f"    {source_domain} --> {target_domain}")

        # 스타일
        lines.append("")
        lines.append("    %% Styles")
        colors = ["#E17055", "#00B894", "#0984E3", "#6C5CE7", "#FDCB6E"]
        for i, domain_id in enumerate(self.parser.domains.keys()):
            color = colors[i % len(colors)]
            lines.append(
                f"    style {domain_id} fill:{color},stroke:#333,stroke-width:2px,color:#fff"
            )

        return "\n".join(lines)


class GraphvizGenerator:
    """Graphviz DOT 형식 생성기 (이미지 생성용)"""

    def __init__(self, parser: LearningGraphParser):
        self.parser = parser

    def generate_full_graph(self) -> str:
        """전체 그래프 DOT 코드 생성"""
        lines = [
            "digraph LearningGraph {",
            "    rankdir=TB;",
            '    node [shape=box, style="rounded,filled", fontname="Arial"];',
            '    edge [fontname="Arial"];',
            "    compound=true;",
            "",
        ]

        # 도메인별 서브그래프
        for domain_id, domain_info in self.parser.domains.items():
            lines.append(f"    subgraph cluster_{domain_id} {{")
            lines.append(f'        label="{domain_info["name"]}";')
            lines.append("        style=rounded;")
            lines.append('        bgcolor="#f5f5f5";')
            lines.append("")

            domain_nodes = [
                n for n in self.parser.nodes.values() if n.domain == domain_id
            ]
            for node in domain_nodes:
                safe_id = node.id.replace(":", "_")
                color = self.parser.categories.get(node.category, {}).get(
                    "color", "#888888"
                )
                lines.append(
                    f'        {safe_id} [label="{node.name}", fillcolor="{color}", fontcolor="white"];'
                )

            lines.append("    }")
            lines.append("")

        # 엣지
        lines.append("    // Edges")
        for edge in self.parser.edges:
            source_safe = edge.source.replace(":", "_")
            target_safe = edge.target.replace(":", "_")
            if edge.edge_type == "external":
                lines.append(
                    f'    {source_safe} -> {target_safe} [style=dashed, color="#666666"];'
                )
            else:
                lines.append(f"    {source_safe} -> {target_safe};")

        lines.append("}")
        return "\n".join(lines)

    def generate_domain_graph(self, domain_id: str) -> str:
        """특정 도메인 그래프 DOT 코드"""
        domain_info = self.parser.domains.get(domain_id, {})

        lines = [
            f"digraph {domain_id} {{",
            "    rankdir=TB;",
            '    node [shape=box, style="rounded,filled", fontname="Arial"];',
            '    edge [fontname="Arial"];',
            f'    label="{domain_info.get("name", domain_id)}";',
            "    labelloc=t;",
            "    fontsize=20;",
            "",
        ]

        # 카테고리별 서브그래프
        domain_nodes = [n for n in self.parser.nodes.values() if n.domain == domain_id]
        categories_in_domain = set(n.category for n in domain_nodes)

        for cat_id in categories_in_domain:
            cat_info = self.parser.categories.get(cat_id, {})
            cat_name = cat_info.get("name", cat_id.split(":")[-1])
            safe_cat = cat_id.replace(":", "_")

            lines.append(f"    subgraph cluster_{safe_cat} {{")
            lines.append(f'        label="{cat_name}";')
            lines.append("        style=rounded;")
            lines.append('        bgcolor="#fafafa";')

            for node in domain_nodes:
                if node.category == cat_id:
                    safe_id = node.id.replace(":", "_")
                    color = cat_info.get("color", "#888888")
                    lines.append(
                        f'        {safe_id} [label="{node.name}", fillcolor="{color}", fontcolor="white"];'
                    )

            lines.append("    }")
            lines.append("")

        # 외부 의존성 노드
        external_nodes = set()
        for edge in self.parser.edges:
            if edge.target_domain == domain_id and edge.source_domain != domain_id:
                external_nodes.add(edge.source)

        if external_nodes:
            lines.append("    // External dependencies")
            for ext_id in external_nodes:
                safe_id = ext_id.replace(":", "_")
                if ext_id in self.parser.nodes:
                    name = self.parser.nodes[ext_id].name
                    ext_domain = ext_id.split(":")[0]
                    domain_name = self.parser.domains.get(ext_domain, {}).get(
                        "name", ext_domain
                    )
                    label = f"{name}\\n({domain_name})"
                    lines.append(
                        f'    {safe_id} [label="{label}", fillcolor="#cccccc", style="rounded,filled,dashed"];'
                    )
            lines.append("")

        # 엣지
        lines.append("    // Edges")
        for edge in self.parser.edges:
            if edge.target_domain == domain_id:
                source_safe = edge.source.replace(":", "_")
                target_safe = edge.target.replace(":", "_")
                if edge.edge_type == "external":
                    lines.append(
                        f'    {source_safe} -> {target_safe} [style=dashed, color="#999999"];'
                    )
                else:
                    lines.append(f"    {source_safe} -> {target_safe};")

        lines.append("}")
        return "\n".join(lines)
