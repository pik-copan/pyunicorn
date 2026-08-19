---
title: >
  `pyunicorn` v1.0: A versatile toolbox for complex system analysis
  across disciplines

tags:
  - complex systems
  - complex networks
  - nonlinear dynamics
  - time series analysis
  - surrogate models
  - Python

authors:
  - given-names: Fritz
    surname: Kühlein
    orcid: 0009-0006-7513-0688
    corresponding: true
    affiliation: "1, 2, 3 *"
  - given-names: Boyan
    surname: Beronov
    orcid: 0000-0002-0900-752X
    affiliation: "4 *"
  - given-names: Max
    surname: Bechthold
    orcid: 0009-0007-7113-4814
    affiliation: "1, 2, 3"
  - given-names: Reik V.
    surname: Donner
    orcid: 0000-0001-7023-6375
    affiliation: "5, 6"
  - given-names: Jonathan F.
    surname: Donges
    orcid: 0000-0001-5233-7703
    affiliation: "1, 2, 7"

affiliations:
  - index: "1"
    name: >
      Earth Resilience Science Unit,
      Potsdam Institute for Climate Impact Research
      (PIK) -- Member of the Leibniz Association,
      Germany
    ror: "03e8s1d88"
  - index: "2"
    name: >
      Integrative Earth System Science,
      Max Planck Institute of Geoanthropology,
      Jena, Germany
    ror: "00js75b59"
  - index: "3"
    name: >
      Institute of Physics and Astronomy,
      University of Potsdam,
      Germany
    ror: "03bnmw459"
  - index: "4"
    name: >
      Computer Science Department,
      University of British Columbia,
      Vancouver, Canada
    ror: "03rmrcq20"
  - index: "5"
    name: >
      Department of Water, Environment, Construction and Safety,
      Magdeburg-Stendal University of Applied Sciences,
      Germany
    ror: "04vjfp916"
  - index: "6"
    name: >
      Research Department IV - Complexity Science,
      Potsdam Institute for Climate Impact Research
      (PIK) -- Member of the Leibniz Association,
      Germany
    ror: "03e8s1d88"
  - index: "7"
    name: >
      Stockholm Resilience Center,
      Stockholm University,
      Sweden
    ror: "0145rpw38"
  - index: "*"
    name:
      _These authors contributed equally to this work._

date: 10 July 2026

bibliography: paper.bib
---

\def\sectionautorefname{Section}


# 1. Summary

The `pyunicorn` (Unified Complex Network and Recurrence Analysis) toolbox
provides a unique collection of data-driven assessment methods for complex
system phenomena --- including various aspects of dynamical complexity,
spatio-temporal interrelations, extreme events and critical transitions. Its
algorithms combine network theory and nonlinear time series analysis, with a
focus on constructing and quantifying the following types of numerical objects:

- **Time series networks** (e.g., recurrence networks, visibility graphs):
  Within a time series, a pairwise relation is computed over
  states/events/patterns, which are in turn defined by a predicate over time
  intervals.
- **Functional networks** (e.g., climate networks):
  Within a collection of mutually dependent time series, members are linked
  according to some measure of statistical association.
- **Surrogate networks/time series** (e.g., twin surrogates):
  Samples are drawn from a conditionally uniform distribution over networks/time
  series, under some structural constraints motivated by hypothesis testing
  purposes.

![
Example of a recurrence network in climatology [@marwan_palaeo_2021][^1].
(**A**) Time series of January insolation at latitude $20 \text{ °N}$.
(**B**) Delay coordinate embedding for **A**, with embedding dimension $m = 2$
and delay $\tau = 6 \text{ ka}$.
(**C**) Recurrence plot for **B**, with recurrence threshold
$\epsilon=10 \text{ W}/\text{m}^2$.
(**D**) Recurrence network for **C**, with nodes of darker colour representing
later points in time.
](img/recurrence_network_steps.pdf){#fig:recnet width=99.5%}

`pyunicorn` is implemented in `Python` and `C/Cython` [@behnel_cython_2011], and
was first published over a decade ago [@donges_unified_2015]. Since then, it has
been maintained as an open source community project, benefitting from numerous
user requests, new features, tutorials, corrections and refactorings. A growing
interdisciplinary user base, as well as fluctuating levels of funding and
development over the years, have warranted a recent concerted effort to
consolidate the package. The present paper accompanies the resulting first major
version release[^2], and compiles applications published to date.

[^1]: Reprinted from @marwan_palaeo_2021 with permission from Elsevier.
Original under [CC BY-NC-ND 4.0](
https://creativecommons.org/licenses/by-nc-nd/4.0/).
[^2]: Cf. [Semantic Versioning specification](https://semver.org).


# 2. Statement of need

Network theory and dynamical system theory have long provided two complementary
perspectives on complex systems: The former analyses the structure of
interactions (links/edges) between subsystems (nodes/vertices), whereas the
latter characterises systemic behaviour in time, such as predictability and
chaos, bifurcations and regime shifts. When the development of `pyunicorn` began
in 2008, these two fields had started growing together specifically along two
strands of research.

On the one hand, once a time series observing a complex system has been
represented as a **time series network**, dynamical properties of the system can
be investigated using network theory methods. Various linking criteria have been
proposed for the construction of time series networks, including recurrence
relations within trajectories in phase space or embedding space [@marwan_2007;
@marwan_rna_2009; @donges_rna_2012] (cf. \autoref{fig:recnet}), and visibility
relations within function graphs of scalar time series
[@lacasa_visibilitygraph_2008].

On the other hand, linear or nonlinear statistical association measures from the
time series analysis literature can be used to construct a **functional
network** from a collection of time series, which yields a topological
description of functional interdependence. This methodology was especially put
forward in neuroscience [@zhou_brain_2006; @zhou_brain_2007;
@bullmore_brain_2009] and climatology [@donges_cn_2009; @donges_cn_eigen_2015],
and has found further applications in fields such as economics and finance
[@huang_stock_2009].

In addition to accompanying these syntheses, the `pyunicorn` library has also
facilitated the development of a number of more specialised methods by
affiliated research groups; and for many such methods, it remains the only
actively maintained open source implementation to date. The emergence of many
new interdisciplinary applications (cf. \autoref{sec:impact}) has further
affirmed the claim that, beyond its origin in climate and Earth system science,
`pyunicorn`'s network approach is "widely applicable in numerous fields"
[@donges_unified_2015].


# 3. State of the field

Long-established packages in the `Python` ecosystem, such as
[`networkx`](https://networkx.org) [@hagberg_networkx_2008],
[`python-igraph`](https://python.igraph.org) [@csardi_Igraph_2006] and
[`networkit`](https://networkit.github.io/) [@angriman_networkit_2022],
extensively cover graph algorithms. In addition,
[`graph-tool`](https://graph-tool.skewed.de/) [@peixoto_graph-tool_2014;
@peixoto_inference_2023] implements nonparametric Bayesian methods for
hierarchical community detection. Libraries with a stronger methodological focus
include: [`PyRQA`](https://pypi.org/project/PyRQA/) [@rawald_pyrqa_2017] and
[`AccRQA`](https://github.com/KAdamek/AccRQA) [@adamek_accrqa_2026] for
recurrence quantification analysis on large datasets,
[`ordpy`](https://ordpy.readthedocs.io/) [@pessa_ordpy_2021] for time series
analysis with ordinal networks, [`smt`](https://smt.readthedocs.io/en/stable/)
[@saves_smt_2024] and
[`irreversibility`](https://pypi.org/project/irreversibility/)
[@zanin_irreversibility_2025] for surrogate modelling and for irreversibility
tests of time series, and
[`pynamicalsys`](https://pypi.org/project/pynamicalsys/)
[@sales_pynamicalsys_2025] for dynamical system analysis.
@siebert_timeseries_2021 provide a systematic review of `Python` packages for
various time series analysis applications. Of course, comparable software also
exists for other programming languages, including: [`CRP Toolbox`](
https://tocsy.pik-potsdam.de/CRPtoolbox/) [@crptoolbox] and
[`NoLiTiA`](https://www.nolitia.com/) [@weber_nolitia_2022] in `Matlab`,
[`DynamicalSystems.jl`](
https://juliadynamics.github.io/DynamicalSystemsDocs.jl/)
[@datseris_DynamicalSystemsjl_2018] in `Julia`, and
[`nonlinearTseries`](https://constantino-garcia.r-universe.dev/nonlinearTseries)
[@garcia_nonlinearTseries_2026] in `R`.

Yet, `pyunicorn` has maintained its unique position, building on established
graph packages to bridge and complement the above application areas. For
instance, `pyunicorn.Network` is implemented by extending `python-igraph.Graph`
with more advanced constructs, such as *coupled* or *multilayer* networks
[@donges_coupled_2011] and *node-weighted* or *node-splitting-invariant* network
measures [@heitzig_nsi_2012; @wiedermann_interacting_nsi_2013;
@zemp_dirweigh_nsi_2014]. Overall, `pyunicorn`'s strength lies in its
integration of methods from complex network theory and nonlinear time series
analysis. @zou_networks_timeseries_2019, @silva_timeseries_networks_2021 and
@marwan_palaeo_2021 provide thorough summaries of relevant concepts and
algorithms.


# 4. Software design

![
Overview of `pyunicorn`’s modules/classes, with an example class inheritance
relation.
](img/module_overview.pdf){#fig:modules width=95%}

`pyunicorn` follows an object-oriented design, with module and class hierarchies
that isolate core data structures and reflect relationships between analysis
methods. For example, `RecurrenceNetwork` inherits much of its functionality
from `RecurrencePlot` and `Network` (cf. \autoref{fig:modules}).

`pyunicorn.Network` is the central class, and it is backed both by a
`python-igraph` graph [@csardi_Igraph_2006] and by a `scipy.sparse` matrix
[@virtanen_scipy_2020]. In addition to importing, constructing, querying,
updating and exporting these internal data structures, `Network` also implements
more specialised network annotations, network measures, and random graph models.
Most other classes in the package derive from `Network`, extending it with
dedicated logic for analyses. Many library methods are implemented by calling
into `pyunicorn`’s own `C/Cython` extension modules via array interfaces.

Overall, this design covers many common use cases in complex system science: It
supports data imports from various formats, its analysis methods are easy to
navigate, and it integrates naturally with the `Python` numerical computing
ecosystem. While out-of-core and high-performance computing were not primary
design goals, `pyunicorn`'s graph algorithms achieve a performance level that is
comparable to, or competitive with, many other general-purpose implementations.

The library has tracked changing standards in package management, extension
compilation, linting, testing and continuous integration, and it is compatible
with all officially supported `Python` versions on Windows, macOS and Linux. For
improved maintainability, the `Python`/`Cython` interface was overhauled, and
most extension modules were ported from `C` to `Cython`. Substantial additions
to the test suite, as well as removal of experimental and untested code, have
increased test coverage from less than 50% to nearly 80%.

`pyunicorn` has also grown by a range of new or improved functionality (cf.
[Changelog](
https://github.com/pik-copan/pyunicorn/blob/joss-paper/CHANGELOG.rst)):

- Many measures in `Network`, and especially their node-splitting-invariant
  refinements, were generalised to support weighted and directed networks.
- Spatial and coupled network analyses were generalised, by adding
  `SpatialNetwork`, by adding a Watts-Strogatz model to `Network`, and by adding
  new measures to `RecurrencePlot`, `CoupledClimateNetwork` and
  `InteractingNetwork`.
- `EventSeries` was added along with `EventSeriesClimateNetwork`, and
  subsequently received several algorithmic improvements.
- `MapPlot` was refactored and simplified.
- State management and memoisation were consolidated via the new mix-in
  `Cached`.
- A collection of tutorial notebooks was added to the documentation.


# 5. Research impact statement
\label{sec:impact}

Since its initial release, `pyunicorn` has found numerous applications across
the sciences[^3]:

- Astrophysics, e.g., active galactic nuclei [@phillipson_galactic_2023], black
  holes [@broadbent_cygnus_2023] and stars [@george_betelgeuse_2020].
- Geophysics, e.g., crustal displacements [@hobbs_gnss_2018], seismic tomography
  [@talavera_mantle_2023] and magnetospheric dynamics
  [@donner_magnetosphere_2019].
- Climate and paleoclimate science [@sumit_india_2026; @jiang_tibet_2024;
  @moinat_tipping_2024; @haas_pitfalls_2023; @sun_texas_2018; @li_pakistan_2026;
  @marwan_palaeo_2021; @caesar_amoc_2020; @franke_anomalies_2017;
  @franke_holocene_2017; @wolf_baiu_2021; @wolf_itcz_2021;
  @wolf_connectivity_2021; @donges2015b; @nocke_visual_2015;
  @wiedermann_hierarchical_2017; @lekscha_windowed_2020], and in particular El
  Niño related phenomena [@feng_enso_2017; @oluwole_enso_2017;
  @broni-bedaiko_enso_2019; @ekhtiari_enso_2021; @wiedermann_enso_2016].
- Social science, e.g., societal clustering and marginalisation
  [@schleussner_minorities_2016].
- Biological and medical research, e.g., mammal evolution
  [@bekeraite_evolutionary_2025], cancer research [@amemiya_cancer_2025],
  behavioural neuroscience [@maksimenko_neural_2019] and EEG data
  [@frolov_recurrence_2023; @lechner_depressive_2024].
- Engineering, e.g., warning signals for thermoacoustic instabilities
  [@waxenegger_rocket_2021] and functional dependence in coupled oscillators
  [@geier_oscillators_2024].
- Particle physics, e.g., collision event analysis [@mullin_susy_2021].
- Methodological advances in network theory and nonlinear dynamics
  [@silini_entropy_2021; @yuan_correlations_2024; @medrano_radius_2021;
  @wiedermann_surrogates_2016; @odenweller_paired-event_2020;
  @sales_stickiness_2023; @subramaniyam_signatures_2015;
  @lekscha_phasespace_2018; @alberti_magnetic_2020].

Furthermore, `pyunicorn`’s tutorial notebooks provide introductions to various
complex system analysis methodologies, and have been repeatedly employed in
academic teaching.

[^3]: 26 out of the 46 surveyed publications are unaffiliated with the package
    authors.


# AI usage disclosure

No generative AI was used in the development and maintenance of this software,
or in the writing of this manuscript. The use of AI coding assistance is not
prohibited for contributors.


# Author contributions

*Software:*
FK coordinated maintenance and releases in recent years. BB contributed to
design and maintenance, provided technical guidance, and edited tutorials. MB
contributed to maintenance and wrote tutorials. JFD and RVD organised funding
and supervised development.
*Manuscript:*
FK and BB developed the full draft. MB drafted \autoref{sec:impact}. MB, JFD and
RVD provided reviews and edits.


# Acknowledgements

We are thankful for all code contributions and bug reports since `pyunicorn`'s
initial release (cf. [Contributions](
https://github.com/pik-copan/pyunicorn/blob/joss-paper/CONTRIBUTIONS.rst)), and
especially for development and maintenance work by Wolfram Barfuss, Guruprem
Bishnoi, Nils Harmening, Ronja Hotz, Johannes Kassel, Jonathan Kroenke and Lena
Schmidt. Recent development was financially supported by the German Federal
Ministry of Education and Research (BMBF) within the scope of the projects
GOTHAM (grant no. 01LP1611A) and ROADMAP (grant no. 01LP2002B). We are grateful
to Jobst Heitzig and Norbert Marwan for their availability for consultation.
Lastly, FK would like to thank Jakob Harteg and Lorenz Sieben for inspiration on
scientific software development in `Python`.


# References
