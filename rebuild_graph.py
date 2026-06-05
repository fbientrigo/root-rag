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

path = Path('.')
detect_result = detect(path)
print(f"Detected {detect_result['total_files']} files")

# AST Extraction
code_files = []
for f in detect_result.get('files', {}).get('code', []):
    code_files.extend(collect_files(Path(f)) if Path(f).is_dir() else [Path(f)])

ast_result = extract(code_files) if code_files else {'nodes':[], 'edges':[]}
print(f"AST: {len(ast_result['nodes'])} nodes")

# Semantic Extraction (Using cache ONLY to avoid LLM calls for now)
all_files = [f for files in detect_result['files'].values() for f in files]
cached_nodes, cached_edges, cached_hyperedges, uncached = check_semantic_cache(all_files)
print(f"Cache: {len(cached_nodes)} nodes from {len(all_files)-len(uncached)} files")

# Merge
seen = {n['id'] for n in ast_result['nodes']}
merged_nodes = list(ast_result['nodes'])
for n in cached_nodes:
    if n['id'] not in seen:
        merged_nodes.append(n)
        seen.add(n['id'])

merged_edges = ast_result['edges'] + cached_edges
merged_hyperedges = cached_hyperedges

# Build Graph
extraction = {
    'nodes': merged_nodes,
    'edges': merged_edges,
    'hyperedges': merged_hyperedges
}

G = build_from_json(extraction)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

if G.number_of_nodes() > 0:
    communities = cluster(G)
    cohesion = score_all(G, communities)
    labels = {cid: f'Community {cid}' for cid in communities}
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    questions = suggest_questions(G, communities, labels)
    
    report = generate(G, communities, cohesion, labels, gods, surprises, detect_result, {'input':0, 'output':0}, '.', suggested_questions=questions)
    Path('graphify-out/GRAPH_REPORT_NEW.md').write_text(report)
    to_json(G, communities, 'graphify-out/graph_full.json')
    to_html(G, communities, 'graphify-out/graph_full.html')
    print("Rebuild complete. Files written to graphify-out/graph_full.*")
else:
    print("No nodes found, check extraction.")
