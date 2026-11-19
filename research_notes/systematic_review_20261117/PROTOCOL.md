# Search and screening protocol

Cutoff: **17 November 2026**, Asia/Calcutta. No lower publication-year limit. The protocol was documented and refined during the search; it was not preregistered. This is a **systematic scoping and novelty review**, not an exhaustive full-text review or a PRISMA-compliant meta-analysis.

## Questions

1. Does prior work compare scalar/channel-wise decay while controlling initial realized decay in delta-rule recurrent memory?
2. Does it separate remembering associations from updating the latest value and preserving unrelated associations?
3. Has it separated channel heterogeneity from token dependence with static/shuffled/averaged gate controls?
4. Are the training-dynamics, interference, or obsolete-answer findings already established?

Population: artificial sequence models with recurrent/associative state. Interventions: gate granularity, conditioning, initialization, decay, write/erase, or controlled training coverage. Outcomes: associative retrieval, latest-value accuracy, retention, stale errors, learning dynamics, and generalization. Comparators: matched architectures/recipes, static/dynamic controls, and attention where relevant.

## Discovery

- Native OpenAlex API: twelve keyword searches, first twenty results per query, cutoff filter. Exact requests, retrieval times, returned counts, and broad reported totals are in `catalog_query_log.json`.
- Native Crossref API: four bibliographic searches, first twenty results each. These return substantial lexical noise; their total-result estimates are not paper counts screened.
- Web-indexed primary sites: arXiv, OpenReview/TMLR, PMLR, NeurIPS, ACL Anthology, author pages, and linked official code. Exact query families and primary follow-up appear in the thematic notes and `web_search_log.json`.
- Site-restricted ACM Digital Library and IEEE Xplore discovery attempts did not establish native-index coverage. Do not describe this review as a complete search of either database.
- Initial API access probes: Semantic Scholar HTTP 429; DBLP HTTP 500. No successful native searches were completed for these services. No Google Scholar, Scopus, or Web of Science export was obtained.

## Citation chasing

Backward chasing followed relevant references in the closest papers and the historical chain described in the thematic notes. Full catalog reference-ID lists were retained where returned; most references were not individually read.

Forward chasing used OpenAlex. The final DOI-verified extension retrieved all indexed citing records up to the cutoff for GDN (343), Kimi Linear (311), and Zoology (167). These numbers describe OpenAlex's returned index at retrieval time, not all citations worldwide. Their metadata and page requests are preserved in `verified_citation_seeds.json`, `citation_extension_log.json`, and `citation_extension_raw.json`.

The initial fuzzy-search pass mistakenly chose two first-ranked records that were not the intended Kimi/Das titles. Its logs are preserved rather than silently rewritten: deepfake/Kimi and ATMA citation searches are **rejected intended-seed matches**, not searches of the intended papers. The DOI-verified extension corrects Kimi and completes GDN/Zoology pagination. Das was reviewed through primary TMLR/paper/code sources; its complete indexed citing set was not obtained. The search script now requires an exact seed-title match before chasing.

Initial DAMI and How Linear Attention Remembers catalog searches returned zero citing records; KATA returned four. These are index observations, never evidence that no later related work exists. Chasing was one-hop and relevance-guided, not recursive traversal of the whole literature.

## Deduplication and screening

Catalog titles were normalized to lowercase alphanumeric keys. The first discovery pass contained 405 distinct normalized-title records; the final combined catalog contains **954**. These are not 954 full papers read. Title normalization does not perfectly merge conference/preprint versions or changed titles. Stable arXiv IDs/DOIs were used for the selected primary evidence register.

Titles were triaged for scope, with primary-source follow-up prioritized for direct claims and close competitors. `screening_ledger.csv` explicitly separates primary-followed-up records, clear out-of-scope title matches, and plausible/background records not selected for primary reading. The latter remain a coverage limitation, not proven irrelevant papers. Automated title tagging assists the ledger and does not replace full-text eligibility assessment.

Retain primary sources about memory recurrence, associative tasks, gate design, initialization, intervention, training dynamics, or capacity that materially constrain a claim. Include older foundations as background even when not direct duplicates. Exclude clear lexical false positives (biology, chemistry, communication networks, unrelated physical gates), unrelated downstream applications, and duplicate implementation-only records from paper evidence. Keep inaccessible close-looking candidates explicitly unresolved.

Evidence levels: **A** = relevant full-text sections inspected; **B** = primary abstract/proceedings/metadata plus scoped interpretation; **C** = catalog-only or inaccessible, no substantive result used. A source's title verification is not full-text verification. `primary_metadata_verification.json` cross-checks forty arXiv title/ID pairs, but the thematic notes determine actual reading depth.

## Bias and limits

This was an AI-assisted review with parallel thematic checks, followed by a shared synthesis; it is not independent blinded human dual screening. English queries dominate. Search ranking, top-result limits, catalog lag, absent subscription exports, access blocks, and undiscovered synonyms can hide relevant work. Several 2026 sources are preliminary preprints; reported findings were not independently reproduced here. No numerical meta-analysis, pooled effect, or publication guarantee is justified.

## Stopping rule and update

Complete the predefined keyword families, historical tracing, bounded forward sets, and primary checks for newly identified close competitors. Stop after the final targeted follow-up round; retain background candidates and unresolved sources as declared gaps. Before submission, rerun the saved queries, check revisions of closest papers, obtain unresolved manuscripts where possible, and repeat focused searches for the final exact claim. A database-sized catalog is not a certificate of priority.
