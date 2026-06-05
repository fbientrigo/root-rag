# Quick Start

This guide provides the bare essentials to get `root-rag` installed and running.

## 1. Installation

First, clone the repository and install the necessary dependencies. This project uses Python 3.10+.

```bash
# Clone the repository
git clone https://github.com/fbientrigo/root-rag
cd root-rag

# Install in editable mode
pip install -e .
```

## 2. Indexing the Default ROOT Corpus

Before you can search, you need to build a search index from the ROOT source code. The following command downloads the necessary source files for ROOT v6.36.08 and builds a small, default index. This is a one-time operation.

```bash
# Build the default index (uses a small 'seed' corpus)
root-rag index
```

This command will fetch the code, process it, and create an index in the `data/indexes` directory.

## 3. Querying the Index

Once the index is built, you can ask questions. The `ask` command performs a lexical search for your keywords.

```bash
# Ask a question about the indexed code
root-rag ask "TTree::Fill"
```

You should see evidence snippets from the ROOT source code, including file paths and line numbers.

## 4. Checking Available Indices

You can see which indices are available and some of their metadata.

```bash
# List all available indices
root-rag versions
```

## 5. Next Steps

You've successfully set up `root-rag`!

-   To learn about more advanced indexing and searching, see the guides in `docs/guides/`.
-   For a complete list of commands, see the `README.md` or run `root-rag --help`.
