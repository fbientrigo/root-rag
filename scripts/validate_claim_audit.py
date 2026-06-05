import json
import sys
import os
import re
from typing import List, Optional
try:
    from pydantic import BaseModel, ValidationError, Field, field_validator
except ImportError:
    print("Error: pydantic is not installed in the current environment.")
    sys.exit(1)

# Define the model to mirror schemas/claim_audit.schema.json
class Claim(BaseModel):
    claim_id: str = Field(pattern=r"^CLM-[A-Z0-9-]+$")
    claim_text: str
    claim_state: str
    evidence_type: str
    profile: str
    source: str
    line_range_or_section: str
    runtime_validated: bool
    physics_validated: bool
    thesis_safe_sentence: str
    remaining_uncertainty: str

    @field_validator('claim_state')
    @classmethod
    def validate_claim_state(cls, v: str) -> str:
        allowed = [
            "CONFIRMED code-local",
            "CONFIRMED external-doc",
            "CONFIRMED project-local-docs",
            "PROVISIONAL",
            "UNRESOLVED",
            "CONTRADICTED"
        ]
        if v not in allowed:
            raise ValueError(f"claim_state must be one of {allowed}")
        return v

    @field_validator('evidence_type')
    @classmethod
    def validate_evidence_type(cls, v: str) -> str:
        allowed = [
            "code-local",
            "external-doc",
            "project-local-docs",
            "mixed",
            "unsupported"
        ]
        if v not in allowed:
            raise ValueError(f"evidence_type must be one of {allowed}")
        return v

class ClaimAudit(BaseModel):
    audit_type: str
    audit_date: str
    fairship_commit: Optional[str] = None
    claims: List[Claim]

def validate_file(file_path: str):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return False

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        ClaimAudit(**data)
        print(f"PASS: {file_path} is valid according to the Claim Audit schema.")
        return True
    except json.JSONDecodeError as e:
        print(f"FAIL: {file_path} is not a valid JSON file. {e}")
        return False
    except ValidationError as e:
        print(f"FAIL: {file_path} failed validation.")
        print(e)
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_claim_audit.py <file1.json> <file2.json> ...")
        sys.exit(1)

    success = True
    for arg in sys.argv[1:]:
        if not validate_file(arg):
            success = False
    
    if not success:
        sys.exit(1)
