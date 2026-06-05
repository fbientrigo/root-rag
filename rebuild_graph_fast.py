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
    print(f"Processing {d}...")
    dr = detect(p)
    for k in all_detect_files:
        all_detect_files[k].extend(dr['files'].get(k, []))
    
    # AST
    code_files = []
    for f in dr['files'].get('code', []):
        code_files.extend(collect_files(Path(f)) if Path(f).is_dir() else [Path(f)])
    if code_files:
        ast = extract(code_files)
        all_nodes.extend(ast['nodes'])
        all_edges.extend(ast['edges'])

# Semantic from cache
flat_files = [f for files in all_detect_files.values() for f in files]
cached_nodes, cached_edges, cached_hyperedges, uncached = check_semantic_cache(flat_files)
print(f"Cache hit: {len(flat_files)-len(uncached)} files")

seen = {n['id'] for n in all_nodes}
for n in cached_nodes:
    if n['id'] not in seen:
        all_nodes.append(n)
        seen.add(n['id'])
all_edges.extend(cached_edges)

# Build
extraction = {'nodes': all_nodes, 'edges': all_edges, 'hyperedges': cached_hyperedges}
G = build_from_json(extraction)
print(f"Final Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

if G.number_of_nodes() > 0:
    communities = cluster(G)
    cohesion = score_all(G, communities)
    labels = {cid: f'Community {cid}' for cid in communities}
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    questions = suggest_questions(G, communities, labels)
    
    report = generate(G, communities, cohesion, labels, gods, surprises, {'total_files': len(flat_files), 'files': all_detect_files, 'total_words': 0}, {'input':0, 'output':0}, '.', suggested_questions=questions)
    Path('graphify-out/GRAPH_REPORT_NEW.md').write_text(report)
    to_json(G, communities, 'graphify-out/graph_full.json')
    print("Rebuild complete.")
else:
    print("No nodes found.")
