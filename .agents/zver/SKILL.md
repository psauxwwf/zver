---
name: zver
description: Local document search skill for MCP_ZVER tools. Explains when to use each tool, what parameters to pass, and how to interpret returned chunks, file names, scores, and metadata.
---

# ZVER MCP Skill

Use this skill when you need to search the local document collection through the `MCP_ZVER_*` tools.

Goals of this skill:

- choose the correct MCP_ZVER tool quickly;
- understand what to pass as input;
- understand what comes back as output;
- avoid confusing content search, document-name search, and full-document retrieval.

## Quick Tool Selection

Use this decision order:

1. If you first need to know what documents exist at all, call `MCP_ZVER_all_names`.
2. If the user is searching by meaning, topic, question, or content description, call `MCP_ZVER_find_by_text_embed`.
3. If the user knows the exact phrase, code, term, or literal text fragment, call `MCP_ZVER_find_by_text`.
4. If the user knows the document name or part of it and wants that document's content, call `MCP_ZVER_all_by_name`.
5. If the user does not know the exact document name but describes it semantically, and wants the full document, call `MCP_ZVER_all_by_name_embed`.

## Core Difference Between Tools

- `find_by_text_embed` returns the best chunks by semantic similarity inside document text.
- `find_by_text` returns chunks containing an exact substring match.
- `all_by_name` finds documents by exact file-name substring and returns all chunks from matching documents.
- `all_by_name_embed` finds documents by semantic similarity of the document name and returns all chunks from matching documents.
- `all_names` returns only document names, not chunks.

Main rule:

- if you need a short relevant answer about a topic, usually start with `find_by_text_embed`;
- if you need the entire document, use `all_by_name` or `all_by_name_embed`.

## Search Result Format

All search tools except `all_names` return a list of objects with this structure:

```json
[
  {
    "id": "stable_chunk_id",
    "name": "Normalized document name",
    "text": "Chunk text",
    "metadata": {
      "path": "docs/file.docx",
      "name": "file.docx",
      "stem": "file",
      "parent": "docs",
      "suffix": ".docx",
      "size_bytes": 123456,
      "mtime_ns": 1730976230000000000,
      "chunk_index": 0,
      "chunk_count": 144
    },
    "score": 0.91
  }
]
```

### Field Meaning

- `id`: stable chunk identifier.
- `name`: normalized source document name.
- `text`: text of the returned chunk.
- `metadata.path`: original file path.
- `metadata.name`: original file name with extension.
- `metadata.stem`: file name without extension.
- `metadata.parent`: parent directory.
- `metadata.suffix`: file extension.
- `metadata.size_bytes`: file size.
- `metadata.mtime_ns`: file modification time.
- `metadata.chunk_index`: chunk index inside the document.
- `metadata.chunk_count`: total chunk count for that document.
- `score`: match score.

### How To Interpret `score`

- in `find_by_text_embed` and `all_by_name_embed`, `score` is a semantic similarity score;
- in `find_by_text` and `all_by_name`, `score` is usually `1.0`, because those tools do exact filtering, not embedding ranking.

## MCP_ZVER_all_names

### When To Use

Use it when you need to:

- inspect what documents exist in the collection;
- choose a correct document name before calling `all_by_name`;
- check whether a document is present at all.

### Input

No parameters.

### Output

A list of strings:

```json
["Document 1", "Document 2", "Document 3"]
```

### Limits

- the list is built from the indexed collection;
- the current implementation is bounded by the backend query cap, so this is not an unlimited directory listing.

### Example Requests

- "Show me what documents are available"
- "Do we have any document about KSCL"
- "First find the right document name"

## MCP_ZVER_find_by_text_embed

### When To Use

Use it for semantic search inside document content.

Good for:

- natural-language questions;
- topics and descriptions;
- cases where the exact wording inside the document is unknown;
- retrieving the best relevant fragments instead of the whole document.

### When Not To Use

Do not use it as the first choice if:

- the user knows the exact phrase and wants a literal match;
- the user needs the whole document, not the best chunks.

### Input

- `query: str`
  Natural-language semantic query.
- `top_k: int = 5`
  Number of best chunks to return.
- `nprobe: int = 10`
  IVF vector search depth. Higher values can improve recall but make the search heavier.

### Behavior

- empty `query` returns `[]`;
- returns only the best chunks, not the entire document;
- usually the best default tool for answering user questions.

### Example Requests

- "how to install KSCL"
- "procedure for updating virus databases"
- "what to do when malware is detected"
- "administration server backup setup"

### Example Call

```python
MCP_ZVER_find_by_text_embed(query="how to install KSCL", top_k=5, nprobe=10)
```

## MCP_ZVER_find_by_text

### When To Use

Use it for exact substring search inside document text.

Good for:

- literal phrases;
- codes, identifiers, command names;
- known terms;
- searching for exact text that should appear verbatim.

### When Not To Use

Do not use it if:

- the user describes a topic in their own words;
- the wording in the document may differ;
- semantic matching is needed.

### Input

- `query: str`
  Exact string to search for in chunk text.
- `top_k: int = 5`
  Number of matching chunks to return.

### Behavior

- empty `query` returns `[]`;
- search is based on substring `LIKE` matching;
- best for commands, product names, parameter names, and other literal strings.

### Example Requests

- `"mysql_secure_installation"`
- `"/opt/kaspersky/ksc64/sbin/klbackup"`
- `"KSC_all_YYYYMMDD.zip"`
- `"kesl-control --start-task 6 --progress"`

### Example Call

```python
MCP_ZVER_find_by_text(query="mysql_secure_installation", top_k=10)
```

## MCP_ZVER_all_by_name

### When To Use

Use it when the user knows the document name or a clear fragment of it and wants the content of the whole document.

Good for:

- exact or partial document names;
- choosing one specific file;
- cases where you need to analyze the whole matched document.

### When Not To Use

Do not use it if:

- the user does not know the document name;
- only the best relevant fragment is needed;
- semantic matching is required instead of literal file-name matching.

### Input

- `query: str`
  Exact substring to match against the document `name`.

### Behavior

- empty `query` returns `[]`;
- if one document matches, returns all its chunks;
- if several documents match, returns chunks from all matching documents.

### Important Consequence

If a matched document has `chunk_count = 144`, the result may contain 144 objects for just that one document.

### Example Requests

- `"Project KSC for Linux Guide"`
- `"KESL"`
- `"Formulary"`

### Example Call

```python
MCP_ZVER_all_by_name(query="Project KSC for Linux Guide")
```

## MCP_ZVER_all_by_name_embed

### When To Use

Use it when the user describes the desired document semantically but does not know the exact document name, and the full document content is needed.

Good for:

- "find the document about KSCL installation";
- "I need the guide about virus database updates";
- "show the document about actions after malware detection".

### When Not To Use

Do not use it as the first choice for a short topic answer.

Reason:

- it returns all chunks from matched documents, not only the best chunks;
- even one matched document can produce a very large response.

If the goal is a compact topical answer, `find_by_text_embed` is usually better.

### Input

- `query: str`
  Semantic description of the desired document.
- `top_k: int = 5`
  Number of candidate document names to keep after semantic name search.
- `nprobe: int = 10`
  IVF vector search depth for document-name embedding search.

### Important `top_k` Detail

Here `top_k` limits the number of matched document names, not the number of chunks inside each matched document.

That means:

- `top_k=1` can still return 144 chunks if the single matched document has 144 chunks;
- `top_k=3` can return the combined chunk count of three matched documents.

### Example Requests

- "document about KSCL installation"
- "guide for Kaspersky Security Center for Linux"
- "document about updating virus databases through the administration server"

### Example Call

```python
MCP_ZVER_all_by_name_embed(query="document about KSCL installation", top_k=1, nprobe=10)
```

## Recommended Usage Strategies

### Strategy 1: Short Answer To A User Question

1. Call `MCP_ZVER_find_by_text_embed`.
2. Use the top 3-5 best chunks.
3. Build the answer from those chunks.
4. If useful, mention the source document via `metadata.path` and `name`.

### Strategy 2: User Knows The Exact Phrase Or Command

1. Call `MCP_ZVER_find_by_text`.
2. Find literal matches.
3. Use those chunks as exact evidence.

### Strategy 3: User Wants A Specific Document

1. If the name is unknown, call `MCP_ZVER_all_names`.
2. If the name is already known, call `MCP_ZVER_all_by_name`.
3. If the name is unknown but the document can be described semantically, call `MCP_ZVER_all_by_name_embed`.

### Strategy 4: First Find The Document, Then Answer Briefly

1. Call `MCP_ZVER_all_by_name_embed` or `MCP_ZVER_all_names`.
2. Identify the target document.
3. Then, if the goal is not the whole document but a precise answer, call `MCP_ZVER_find_by_text_embed` or `MCP_ZVER_find_by_text` with a narrower query.

## Practical Rules

- Do not start with `all_by_name_embed` if the user simply asks a topical question. It often returns too many chunks.
- If the user clearly needs the whole document, then `all_by_name` or `all_by_name_embed` is appropriate.
- If quotation accuracy matters, use `find_by_text`.
- If semantic relevance matters, use `find_by_text_embed`.
- If you first need to know what is available, use `all_names`.

## How To Explain Results To The User

When answering from chunks:

- mention the document via `name`;
- if useful, mention the original file via `metadata.path`;
- if referencing a specific fragment, `chunk_index` can be useful.

When the result is too large:

- do not dump all chunks blindly;
- summarize first;
- then quote only the most relevant fragments.

## Short Cheat Sheet

- `all_names`: list available documents.
- `find_by_text_embed`: best chunks by semantic content.
- `find_by_text`: exact text match.
- `all_by_name`: full document by exact name fragment.
- `all_by_name_embed`: full document by semantic document name.

If you are unsure between `find_by_text_embed` and `all_by_name_embed`, almost always start with `find_by_text_embed`.
