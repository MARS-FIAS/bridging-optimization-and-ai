# Sources and third-party software

The biological model and stochastic simulation implementation build on the
research software of Michael Alexander Ramirez Sierra. The associated work is
described in [Ramirez Sierra and Sokolowski, PLOS Computational Biology
(2024)](https://doi.org/10.1371/journal.pcbi.1012473), with the cavity extension
developed in the [doctoral thesis (2025)](https://doi.org/10.21248/gups.92001).

The earlier research repositories,
[Article_Modeling_Mouse_ICM](https://github.com/MARS-FIAS/Article_Modeling_Mouse_ICM)
and [Article_AI-MAPE_SA-SGM](https://github.com/MARS-FIAS/Article_AI-MAPE_SA-SGM),
carry the same MIT copyright/permission notice. That notice is preserved verbatim
in [`licenses/original-research-MIT.txt`](licenses/original-research-MIT.txt).
The licensing status of this distribution is recorded in
[`LICENSE_STATUS.md`](LICENSE_STATUS.md).

Third-party packages are installed separately; their source distributions and
binary wheels are not bundled here. Their own licenses apply. In particular,
the pinned neural environment uses **sbi 0.19.2 and pyknos 0.15.1**, whose
distributions identify their license as AGPLv3. This project's licensing does
not replace those dependency terms. See the
[sbi 0.19.2 source](https://github.com/mackelab/sbi/tree/v0.19.2),
[pyknos 0.15.1 source](https://github.com/mackelab/pyknos/tree/v0.15.1)
and the [pinned environment](environments/README.md).

The supplied estimator files contain the released model state and refer to
classes from that separately installed environment. They are not dependency
installers. Checksums, parameter order and loading instructions are in
[`artifacts/`](artifacts/).
