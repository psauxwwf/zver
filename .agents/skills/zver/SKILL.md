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
2. If the user is searching by meaning, topic, question, or content description, call `MCP_ZVER_find_by_text_dense`.
3. If the user needs lexical ranked search over query terms, call `MCP_ZVER_find_by_text_bm25`.
4. If the user wants a balanced semantic + lexical result, call `MCP_ZVER_find_by_text_hybrid`.
5. If the user knows the exact phrase, code, term, or literal text fragment, call `MCP_ZVER_find_by_text_grep`.
6. If the user knows the document name or part of it and wants that document's content, call `MCP_ZVER_all_by_name_grep`.
7. If the user does not know the exact document name but describes it semantically, and wants the full document, call `MCP_ZVER_all_by_name_dense`.

## Core Difference Between Tools

- `find_by_text_dense` returns the best chunks by semantic similarity inside document text.
- `find_by_text_bm25` returns the best chunks by lexical BM25 ranking inside document text.
- `find_by_text_hybrid` combines dense and BM25 retrieval for document text.
- `find_by_text_grep` returns chunks whose text matches a GNU grep pattern.
- `all_by_name_grep` finds documents by GNU grep pattern over file names and returns all chunks from matching documents.
- `all_by_name_dense` finds documents by semantic similarity of the document name and returns all chunks from matching documents.
- `all_names` returns only document names, not chunks.

Main rule:

- if you need a short relevant answer about a topic, usually start with `find_by_text_dense`;
- if exact terms matter but you still want ranked retrieval, use `find_by_text_bm25`;
- if you want one default that balances semantics and term matching, use `find_by_text_hybrid`;
- if you need the entire document, use `all_by_name_grep` or `all_by_name_dense`.

## When To Prefer Each Method

- `find_by_text_dense`: best when the user asks by meaning and the exact wording in the document may differ.
- `find_by_text_bm25`: best when exact query terms matter, especially jargon, acronyms, commands, and rare terms.
- `find_by_text_hybrid`: best default for text search when both meaning and exact terms may matter.
- `find_by_text_grep`: best when GNU grep-style pattern matching is required.
- `all_by_name_grep`: best when the document name is already known.
- `all_by_name_dense`: best when the document name is unknown but its meaning is known.

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

- in `find_by_text_dense` and `all_by_name_dense`, `score` is a semantic similarity score;
- in `find_by_text_bm25`, `score` is a lexical BM25-style relevance score;
- in `find_by_text_hybrid`, `score` is the hybrid reranker score after merging dense and BM25 candidates;
- in `find_by_text_grep` and `all_by_name_grep`, `score` is usually `1.0`, because those tools do exact filtering, not embedding ranking.

## MCP_ZVER_find_by_text_bm25

### When To Use

Use it for lexical ranked search when exact query terms matter and you want BM25-style relevance instead of a raw substring filter.

Good for:

- acronym-heavy queries;
- jargon and rare terms;
- cases where exact query words matter, but you still want ranked results instead of raw substring filtering.

### When Not To Use

Do not use it if:

- the user is asking a broad semantic question in their own words;
- the wording may differ heavily from the source text;
- the user needs the whole document.

### Input

- `query: str`
  Natural-language or keyword query for lexical BM25 search in document content.
- `top_k: int = 5`
  Number of best chunks to return.

### Behavior

- empty `query` returns `[]`;
- returns matching document chunks with BM25-style ranking;
- requires a collection built by the current embed pipeline.

### Example Requests

- "evilginx phishlet"
- "mysql_secure_installation"
- "klbackup"

### Example Call

```python
MCP_ZVER_find_by_text_bm25(query="evilginx phishlet", top_k=10)
```

## MCP_ZVER_find_by_text_hybrid

### When To Use

Use it for combined semantic and lexical retrieval over document content. It mixes dense vector search with BM25-style sparse search and re-ranks the merged candidates.

Good for:

- short practical queries where both meaning and exact terms matter;
- ambiguous queries where dense alone may drift and BM25 alone may be too narrow;
- a strong default when the user wants search quality rather than a specific retrieval style.

### When Not To Use

Do not use it if:

- the user explicitly needs an exact substring match;
- the user needs the whole document rather than top chunks.

### Input

- `query: str`
  Natural-language query for hybrid dense plus BM25 search in document content.
- `top_k: int = 5`
  Maximum number of matching chunks to return after re-ranking.
- `nprobe: int = 10`
  IVF vector-search breadth for the dense branch. Higher values may improve recall at the cost of more work.

### Behavior

- empty `query` returns `[]`;
- mixes dense and BM25 candidates and re-ranks the merged results;
- requires a collection built by the current embed pipeline.

### Example Requests

- "evilginx setup"
- "backup administration server"
- "virus database update"

### Example Call

```python
MCP_ZVER_find_by_text_hybrid(query="evilginx setup", top_k=10, nprobe=10)
```

## MCP_ZVER_all_names

### When To Use

Use it when you need to:

- inspect what documents exist in the collection;
- choose a correct document name before calling `all_by_name_grep`;
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
- the list is built by reading the indexed collection, so very large collections may still be expensive to inspect.

### Example Requests

- "Show me what documents are available"
- "Do we have any document about KSCL"
- "First find the right document name"

## MCP_ZVER_find_by_text_dense

### When To Use

Use for semantic content search when the query describes meaning, topic, or intent and exact wording may differ from the source text.

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
  Natural-language query for semantic search in document content. Use topics, questions, paraphrases, or intent descriptions.
- `top_k: int = 5`
  Maximum number of matching chunks to return. Higher values return more candidates.
- `nprobe: int = 10`
  IVF vector-search breadth. Higher values may improve recall at the cost of more work.

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
MCP_ZVER_find_by_text_dense(query="how to install KSCL", top_k=5, nprobe=10)
```

## MCP_ZVER_find_by_text_grep

### When To Use

Use for GNU grep pattern search when you expect the source text to match an ordinary grep regular expression.

Good for:

- literal phrases;
- basic grep regular expressions;
- codes, identifiers, command names;
- known terms;
- searching for text that should match a grep pattern.

### When Not To Use

Do not use it if:

- the user describes a topic in their own words;
- the wording in the document may differ;
- semantic matching is needed.

Important:

- this is grep pattern syntax, not SQL `LIKE`;
- this is pattern input only, not grep CLI flags like `-i`, `-E`, or `-F`;
- syntax is ordinary GNU grep syntax, so grouping and alternation use escaped forms like `\(` `\)` and `\|`.

### Input

- `query: str`
  GNU grep pattern to match inside chunk text. Best for literal phrases, identifiers, codes, or grep regular expressions.
- `top_k: int = 5`
  Number of matching chunks to return.

### Behavior

- empty `query` returns `[]`;
- search is based on GNU grep regex matching;
- best for commands, product names, parameter names, and other literal strings or grep patterns.

### Grep Syntax Examples

- `evilginx`
  simple literal match.
- `^Chapter [0-9][0-9]*$`
  line that starts with `Chapter ` and a positive integer.
- `mysql_.*`
  `mysql_` followed by any characters.
- `KSC_all_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]\.zip`
  archive name with an 8-digit date and literal `.zip`.
- `\(error\|warning\)`
  match either `error` or `warning` using ordinary grep alternation syntax.

### Example Requests

- `"mysql_secure_installation"`
- `"/opt/kaspersky/ksc64/sbin/klbackup"`
- `"KSC_all_YYYYMMDD.zip"`
- `"kesl-control --start-task 6 --progress"`
- `"^Step [0-9][0-9]*$"`
- `"\\(error\\|warning\\)"`

### Example Call

```python
MCP_ZVER_find_by_text_grep(query="mysql_secure_installation", top_k=10)
```

## MCP_ZVER_all_by_name_grep

### When To Use

Use when the user knows all or part of the file name and wants the full content for every matching source using a GNU grep pattern.

Good for:

- exact or partial document names;
- grep patterns over document names;
- choosing one specific file;
- cases where you need to analyze the whole matched document.

### When Not To Use

Do not use it if:

- the user does not know the document name;
- only the best relevant fragment is needed;
- semantic matching is required instead of literal file-name matching.

Important:

- this is grep pattern syntax, not SQL `LIKE`;
- ordinary GNU grep syntax applies here too.

### Input

- `query: str`
  GNU grep pattern to match against normalized file names. Use when the document name or part of it is already known.

### Behavior

- empty `query` returns `[]`;
- if one document matches, returns all its chunks;
- if several documents match, returns chunks from all matching documents.

### Grep Syntax Examples

- `KESL`
  literal file-name fragment.
- `^Project .* Guide$`
  title that starts with `Project ` and ends with ` Guide`.
- `Formulary\|Guide`
  either `Formulary` or `Guide` in the normalized document name.
- `[Kk]aspersky`
  case-sensitive character class matching either capitalized or lowercase first letter.

### Important Consequence

If a matched document has `chunk_count = 144`, the result may contain 144 objects for just that one document.

### Example Requests

- `"Project KSC for Linux Guide"`
- `"KESL"`
- `"Formulary"`
- `"^Project .* Guide$"`
- `"Formulary\\|Guide"`

### Example Call

```python
MCP_ZVER_all_by_name_grep(query="Project KSC for Linux Guide")
```

## MCP_ZVER_all_by_name_dense

### When To Use

Use when the user refers to a document by approximate meaning or paraphrased title and you need the full contents of the most relevant files.

Good for:

- "find the document about KSCL installation";
- "I need the guide about virus database updates";
- "show the document about actions after malware detection".

### When Not To Use

Do not use it as the first choice for a short topic answer.

Reason:

- it returns all chunks from matched documents, not only the best chunks;
- even one matched document can produce a very large response.

If the goal is a compact topical answer, `find_by_text_dense` is usually better.

### Input

- `query: str`
  Natural-language description of the desired document name or title. Use when the exact file name is unknown.
- `top_k: int = 5`
  Maximum number of candidate file names to keep before returning all chunks for those files.
- `nprobe: int = 10`
  IVF vector-search breadth for semantic file-name matching. Higher values may improve recall.

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
MCP_ZVER_all_by_name_dense(query="document about KSCL installation", top_k=1, nprobe=10)
```

## Recommended Usage Strategies

### Strategy 1: Short Answer To A User Question

1. Call `MCP_ZVER_find_by_text_dense`.
2. Use the top 3-5 best chunks.
3. Build the answer from those chunks.
4. If useful, mention the source document via `metadata.path` and `name`.

### Strategy 1a: Term-Sensitive Ranked Search

1. Call `MCP_ZVER_find_by_text_bm25`.
2. Use the top 3-5 best chunks.
3. Prefer this when the user query contains important exact terms.

### Strategy 1b: Best General Text Retrieval

1. Call `MCP_ZVER_find_by_text_hybrid`.
2. Use the top 3-5 best chunks.
3. Prefer this when both semantics and exact terms may matter.

### Strategy 2: User Knows The Exact Phrase Or Command

1. Call `MCP_ZVER_find_by_text_grep`.
2. Find grep-pattern matches.
3. Use those chunks as exact evidence.

### Strategy 3: User Wants A Specific Document

1. If the name is unknown, call `MCP_ZVER_all_names`.
2. If the name is already known, call `MCP_ZVER_all_by_name_grep`.
3. If the name is unknown but the document can be described semantically, call `MCP_ZVER_all_by_name_dense`.

### Strategy 4: First Find The Document, Then Answer Briefly

1. Call `MCP_ZVER_all_by_name_dense` or `MCP_ZVER_all_names`.
2. Identify the target document.
3. Then, if the goal is not the whole document but a precise answer, call `MCP_ZVER_find_by_text_dense` or `MCP_ZVER_find_by_text_grep` with a narrower query.

## Practical Rules

- Do not start with `all_by_name_dense` if the user simply asks a topical question. It often returns too many chunks.
- If the user clearly needs the whole document, then `all_by_name_grep` or `all_by_name_dense` is appropriate.
- If query wording matters and you still want ranking, use `find_by_text_bm25`.
- If you want the strongest general text-search default, use `find_by_text_hybrid`.
- If quotation accuracy matters, use `find_by_text_grep`.
- If semantic relevance matters, use `find_by_text_dense`.
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
- `find_by_text_dense`: best chunks by semantic content.
- `find_by_text_bm25`: best chunks by lexical BM25 ranking.
- `find_by_text_hybrid`: best chunks by dense + BM25 retrieval.
- `find_by_text_grep`: GNU grep pattern match over text.
- `all_by_name_grep`: full document by GNU grep pattern match over file name.
- `all_by_name_dense`: full document by semantic document name.

If you are unsure between `find_by_text_dense`, `find_by_text_bm25`, and `find_by_text_hybrid`, start with `find_by_text_hybrid` for a compact answer and `all_by_name_dense` only when the whole document is needed.
