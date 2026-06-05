import json
from graphify.detect import detect
from graphify.extract import collect_files, extract
from graphify.cache import check_semantic_cache, save_semantic_cache
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json, to_html
from pathlib import Path
import networkx as nx

# Only look at these directories
target_dirs = ['src', 'reports', 'docs', 'tests', 'configs']
all_nodes = []
all_edges = []
all_detect_files = {'code': [], 'document': [], 'paper': [], 'image': []}

for d in target_dirs:
    p = Path(d)
    if not p.exists(): continue
    dr = detect(p)
    for k in all_detect_files:
        all_detect_files[k].extend(dr['files'].get(k, []))
    code_files = []
    for f in dr['files'].get('code', []):
        code_files.extend(collect_files(Path(f)) if Path(f).is_dir() else [Path(f)])
    if code_files:
        ast = extract(code_files)
        all_nodes.extend(ast['nodes'])
        all_edges.extend(ast['edges'])

flat_files = [f for files in all_detect_files.values() for f in files]
cached_nodes, cached_edges, cached_hyperedges, uncached = check_semantic_cache(flat_files)

seen = {n['id'] for n in all_nodes}
for n in cached_nodes:
    if n['id'] not in seen:
        all_nodes.append(n)
        seen.add(n['id'])
all_edges.extend(cached_edges)

extraction = {'nodes': all_nodes, 'edges': all_edges, 'hyperedges': cached_hyperedges}
G = build_from_json(extraction)

if G.number_of_nodes() > 0:
    gods = god_nodes(G)
    top_5 = gods[:5]
    print(f"Top 5 God Nodes: {[g['label'] for g in top_5]}")
    
    start_nodes = [g['id'] for g in top_5]
    subgraph_nodes = set(start_nodes)
    subgraph_edges = []
    
    frontier = set(start_nodes)
    for d in range(2):
        next_frontier = set()
        for n in frontier:
            for neighbor in G.neighbors(n):
                if neighbor not in subgraph_nodes:
                    next_frontier.add(neighbor)
                    subgraph_edges.append((n, neighbor))
                elif (n, neighbor) not in subgraph_edges and (neighbor, n) not in subgraph_edges:
                    subgraph_edges.append((n, neighbor))
        subgraph_nodes.update(next_frontier)
        frontier = next_frontier
    
    print(f"Exploration depth 2: {len(subgraph_nodes)} nodes, {len(subgraph_edges)} edges")
    for u, v in subgraph_edges[:30]:
        rel = G.edges[u,v].get('relation', '')
        u_label = G.nodes[u].get('label', u)
        v_label = G.nodes[v].get('label', v)
        print(f"  {u_label} --{rel}--> {v_label}")
else:
    print("No nodes found.")
