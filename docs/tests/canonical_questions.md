# canonical_questions

## Purpose
This file defines a first-pass evaluation set for the ROOT RAG system.
The goal is not to lock down exact wording, but to define what counts as a correct grounded answer.

## Grading rules
A result counts as correct when:
- it retrieves at least one highly relevant evidence item for the target symbol or concept
- the evidence belongs to the requested ROOT version
- file paths and line ranges are present
- the answer does not invent unsupported API details

A result is incorrect when:
- it cites the wrong symbol
- it mixes versions without saying so
- it answers without evidence
- it gives a generic explanation with no ROOT grounding

## Symbol-oriented questions

1. **Where is `TTree::Draw` declared?**
   - Correct if it retrieves the relevant `TTree` header declaration for the chosen ROOT revision.

2. **Where is `TTree::Draw` implemented?**
   - Correct if it retrieves the corresponding implementation in a source file.

3. **Where is `RDataFrame` declared or introduced in the codebase?**
   - Correct if it retrieves a header or defining file for `ROOT::RDataFrame` or the relevant namespace/class wrapper.

4. **What file contains the declaration of `TFile`?**
   - Correct if it points to the relevant header for `TFile`.

5. **Where is `TChain` defined?**
   - Correct if it retrieves the class declaration and, ideally, implementation references.

6. **Which file declares `TBranchElement`?**
   - Correct if it points to the matching header for the exact class.

7. **Where is `TH1::Fill` declared?**
   - Correct if it finds the histogram class declaration for the relevant overload family.

8. **Where is `TH1::Fit` declared or documented?**
   - Correct if it finds a declaration or nearby Doxygen / reference block that supports the answer.

9. **Where is `TTreeReader` declared?**
   - Correct if it finds the right class declaration file.

10. **Which file contains `TTreeFormula`?**
    - Correct if it identifies the correct header or implementation file.

11. **Where is `ROOT::EnableImplicitMT` declared?**
    - Correct if it identifies the declaration site and symbol context.

12. **Where is `RVec` declared?**
    - Correct if it retrieves the relevant VecOps header / symbol location.

## Concept-oriented questions

13. **How is lazy evaluation described or implemented in `RDataFrame`?**
    - Correct if it retrieves grounded evidence from docs, comments, or implementation indicating deferred execution semantics.

14. **How does ROOT describe implicit multi-threading?**
    - Correct if it retrieves grounded documentation or source comments tied to the relevant API.

15. **What evidence shows how `TTree` handles branches?**
    - Correct if it retrieves branch-related members, methods, or class relationships from `TTree` and related branch classes.

16. **How are histogram fills represented in `TH1`?**
    - Correct if it retrieves grounded `TH1` methods or comments rather than a generic histogram explanation.

17. **What evidence explains how `TFile` opens files?**
    - Correct if it retrieves declarations, factory/open methods, or source comments relevant to file opening behavior.

18. **How does ROOT document friend trees or tree relationships?**
    - Correct if it retrieves evidence around friend-tree APIs or relevant docs.

19. **What evidence describes tree reading with `TTreeReader`?**
    - Correct if it retrieves declarations, Doxygen, or docs connected to `TTreeReader`.

20. **How are selection expressions handled around `TTreeFormula` or tree draw expressions?**
    - Correct if it retrieves evidence linking formula parsing or expression usage to the relevant classes.

## Versioning and provenance questions

21. **Show evidence for `TTree::Draw` in ROOT `v6-32-00`.**
    - Correct if all evidence comes from that version and the version metadata is explicit.

22. **Compare evidence for `RDataFrame` between two ROOT revisions.**
    - Correct in future version-diff mode only; for MVP it is acceptable to refuse and state that cross-version diff is not yet supported.

23. **What commit is this answer based on?**
    - Correct if the response exposes the resolved commit for the selected corpus.

24. **List available indexed ROOT versions.**
    - Correct if the system reports known versions / indexes and does not invent missing ones.

## Failure-mode questions

25. **Where is `TotallyFakeROOTClass` declared?**
    - Correct if the system returns no evidence and refuses unsupported claims.

26. **What does `TTree::DefinitelyNotAMethod` do?**
    - Correct if the system reports insufficient evidence instead of inventing a method.

27. **Explain a ROOT symbol without citations.**
    - Correct behavior is to refuse or still include citations; uncited technical answers do not count.

28. **Answer using mixed versions without stating it.**
    - Correct behavior is to avoid mixing versions, or explicitly label a multi-version response if that feature exists.

## Stretch questions for later branches

29. **What documentation supports `RVec` usage examples?**
    - Correct if the hybrid retrieval branch finds docs and symbol declarations.

30. **What changed for a symbol between version A and version B?**
    - Correct only after `feat/version-diff` is implemented; before that, the correct behavior is a grounded refusal.
