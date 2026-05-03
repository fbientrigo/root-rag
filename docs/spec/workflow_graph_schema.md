# Workflow Graph Schema

Purpose:
- Represent workflow structure as lightweight local JSON without GraphRAG or graph databases.
- Keep workflow claims bounded by evidence status.

Validation commands:
```bash
python scripts/validate_workflow_graph.py workflow_graphs/muon_dis_workflow.json
```

Schema:
```json
{
  "graph_id": "...",
  "created": "...",
  "scope": "...",
  "nodes": [
    {
      "id": "...",
      "label": "...",
      "kind": "script | function | class | config | data | stage | unknown",
      "status": "CONFIRMED | PROVISIONAL | UNRESOLVED",
      "sources": []
    }
  ],
  "edges": [
    {
      "source": "...",
      "target": "...",
      "relation": "calls | reads | writes | configures | produces | consumes | precedes | unknown",
      "status": "CONFIRMED | PROVISIONAL | UNRESOLVED",
      "sources": [],
      "notes": ""
    }
  ]
}
```

Rules:
- `CONFIRMED` node requires at least one source.
- `CONFIRMED` edge requires at least one source.
- If `CONFIRMED` edge has only one source, `notes` must explain single-source limitation.
- Edge `source` and `target` must reference existing node ids.
- Enum fields must use allowed values.

Source anchor format:
- `path/to/file.ext:start-end`

Scope safety:
- Stage-level graphs are acceptable when file/function anchors are not verified.
- Do not mark detailed script/function nodes as `CONFIRMED` without evidence anchors.
