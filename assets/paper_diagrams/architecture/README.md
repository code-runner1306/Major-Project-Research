# Architecture figure

The paper-ready vector PDF is generated at
`indicon_gate_paper/figures/gdn_architecture.pdf` by:

```bash
python assets/paper_diagrams/architecture/architecture_pdf.py
```

`architecture_pdf.py` uses ReportLab drawing primitives only, so the resulting
PDF contains vector boxes, arrows, and text. Its compact serif typography and
pastel rounded boxes use the camera-ready paper as a style reference.

Shows the four recurrent blocks, LayerNorms, identity state mixer, decay and write gates, and readout for the small synthetic recipe. The recurrent state has value rows and key columns.

The TikZ source remains available as an editable alternative. The generated
PDF is the checked artifact for the current manuscript. The diagram reflects
the pinned recipe model: four `RecipeGDN` sequence-mixer blocks, no learned
position table, `nn.LayerNorm` around each block, and an `nn.Identity` state
mixer from the Zoology backbone. It should not be read as a diagram of every
production GDN or hybrid-attention configuration in the repository.
