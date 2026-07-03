---
title: "`pyunicorn` v1.0: A versatile toolbox for complex system analysis across disciplines"
tags:
  - python
  - complex systems
  - network metrics
  - time series analysis
authors:
  - given-names: Fritz
    surname: Kühlein
    orcid: 0009-0006-7513-0688
    corresponding: true
    affiliation: 1, 2, 3
  - given-names: Boyan
    surname: Beronov
    orcid: 0000-0002-0900-752X
    affiliation: "4"
  - given-names: Max
    surname: Bechthold
    orcid: 0009-0007-7113-4814
    affiliation: 1, 2, 3
  - given-names: Reik V.
    surname: Donner
    orcid: 0000-0001-7023-6375
    affiliation: 5, 7
  - given-names: Jonathan F.
    surname: Donges
    orcid: 0000-0001-5233-7703
    affiliation: 1, 2, 6
affiliations:
  - name: Earth Resilience Science Unit, Potsdam Institute for Climate Impact Research (PIK) - Member of the Leibniz Association, Telegrafenberg A31, D-14473 Potsdam, Germany
    index: 1
  - name: Integrative Earth System Science, Max Planck Institute of Geoanthropology, Kahlaische Str. 10, Jena, 07745, Germany
    index: 2
  - name: Institute of Physics and Astronomy, University of Potsdam, D-14476 Potsdam, Germany
    index: 3
  - name: Computer Science Department, University of British Columbia, Vancouver, BC, Canada
    index: 4
  - name: Department of Water, Environment, Construction and Safety, Magdeburg-Stendal University of Applied Sciences, Breitscheidstraße 2, D-39114 Magdeburg, Germany
    index: 5
  - name: Stockholm Resilience Center, Stockholm University, Albanovägen 28, SE-114 19 Stockholm, Sweden
    index: 6
  - name: Research Department IV - Complexity Science, Potsdam Institute for Climate Impact Research (PIK) - Member of the Leibniz Association, Telegrafenberg A31, D-14473 Potsdam, Germany
    index: 7
date: 10 July 2026
bibliography: paper.bib
---

# Summary

The `pyunicorn` (Unified Complex Network and Recurrence Analysis) toolbox provides a unique collection of algorithms for complex system analysis, and is implemented in Python and C/Cython. It was developed for the data-driven assessment of complex system phenomena across disciplines, such as quantification of dynamical complexity, causal interrelations or teleconnections, as well as tipping dynamics. `pyunicorn` bundles a variety of methods drawn from two commonly distinct branches in complex system science – network theory and nonlinear time series analysis. Within this scope, the methodological focus of the package lies on (1) *network representations of time series* linking states, events or patterns exhibited by individual time series or time series groups (e.g. recurrence networks or visibility graphs), (2) the construction and analysis of *functional networks* among sets of time series (e.g. climate networks), as well as (3) the constrained generation of *surrogate networks and time series* for hypothesis testing (e.g. white-noise time series surrogates).

The package was first published more than a decade ago, with a detailed description provided by @donges_unified_2015. Since then, it has been maintained as an open source community project, with several minor version releases that encompassed numerous feature contributions, tutorials, bug fixes and maintainability improvements. In the present paper, which accompanies the first major version release `pyunicorn` v1.0, we summarize the work leading up to this release and collate the diverse scientific applications that have been published over the past decade.

# Statement of need

Complex network theory and nonlinear time series analysis have long provided two complementary perspectives on the structure and dynamics of complex systems. While the analysis of complex networks has focussed on the structure of interactions (links or edges) between subsystems (nodes or vertices), nonlinear time series analysis assessed dynamical aspects such as predictability, chaos, dynamical transitions or bifurcations in the state variables of complex systems.

In the years of `pyunicorn`’s original development – which started in 2008 – both fields had been growing together specifically along two research strands.

On the one hand, the analysis of *functional networks* applies methods from linear and nonlinear time series analysis to construct networks of statistical interrelationships among collections of time series and, subsequently, studies the resulting functional networks by means of methods from complex network theory. This methodology was especially put forward in neuroscience (functional brain networks) [@zhou_brain_2006; @zhou_brain_2007; @bullmore_brain_2009] and climatology (climate networks) [@donges_cn_2009; @donges_cn_eigen_2015], but found further application fields such as economics and finance [@huang_stock_2009].

On the other hand, *network-based time series analysis* investigates the dynamical properties of complex systems based on uni- or multivariate time series data using methods from network theory. Various types of time series networks have been proposed for performing this type of analysis, including recurrence networks based on the recurrence properties of phase space trajectories [@marwan_2007; @marwan_rna_2009; @donges_rna_2012] (see fig. \ref{recnet}) and visibility graphs representing visibility relationships between data points in a time series [@lacasa_visibilitygraph_2008].

`pyunicorn` has accompanied the emergence of these methodological synergies and provides an architecture that facilitates more specialised – and so far less widely applied –  methods that derive from the synthesis of the two fields. Many of the contained methods were established by the research groups affiliated with `pyunicorn`, and at present, this package hosts their only free and actively maintained implementation.

![Illustration of steps to construct a recurrence network for an example from climatology. (A) Time series of January insolation at latitude 20°N. (B) Its time-embedded phase space representation, with embedding dimension $m = 2$ and delay $\tau = 6$ ka. (C) Recurrence plot of the phase space trajectory with recurrence threshold $\epsilon = 10$. (D) Recurrence network constructed from it, with darker color of nodes representing later points in time. [Adapted from @marwan_palaeo_2021] \label{recnet}](img/recurrence_network_steps.png)

Originally developed and applied in the specific context of climate and Earth system science, `pyunicorn`’s first publication conjectured the generality of the network approach allowing for a much wider applicability [@donges_unified_2015]. Indeed, a multitude of subsequent applications have since emerged from new research avenues in a wide range of disciplines, as reviewed in section \ref{research_impact_statement}. Concurrently, `pyunicorn` itself has expanded over the years, as various scientific and non-scientific users have requested or contributed new functionality. With a growing community and code base, and with fluctuating availability of funding and expertise, it has proven a challenge to maintain the package to a solid standard over time. Recently, a targeted effort tackled a significant maintenance backlog and further consolidated the API, which is now reflected in the release of version 1.0 (cf. [semver.org](https://semver.org)).

# State of the field

Several other Python libraries overlap with various aspects of `pyunicorn`'s spectrum of functionality. Long-established packages such as [`networkx`](https://networkx.org) [@hagberg_networkx_2008], [`python-igraph`](https://python.igraph.org) [@csardi_Igraph_2005] and [`networkit`](https://networkit.github.io/) [@angriman_networkit_2022] extensively cover network and graph calculations. [`PyRQA`](https://pypi.org/project/PyRQA/) is a more specialized tool for recurrence quantification analysis that is optimized for handling large datasets [@rawald_pyrqa_2017], and [`ordpy`](https://pypi.org/project/ordpy/) supports time series analysis with ordinal networks [@pessa_ordpy_2021]. Users seeking surrogate modelling of time series can resort to [`smt`](https://pypi.org/project/smt/) [@bouhlel_smt_2019]. More recently, the packages [`irreversibility`](https://pypi.org/project/irreversibility/) [@zanin_irreversibility_2025] for irreversibility tests of time series and [`pynamicalsys`](https://pypi.org/project/pynamicalsys/) [@sales_pynamicalsys_2025] for dynamical systems analysis were released. A systematic review of Python packages for various time series analysis applications was provided by @siebert_timeseries_2021. Comparable software exists for other programming languages including Matlab [e.g., [`NoLiTiA`](https://www.nolitia.com/), @weber_nolitia_2022], Julia [e.g., [`DynamicalSystems.jl`](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/), @datseris_DynamicalSystemsjl_2018], and R [e.g., [`nonlinearTseries`](https://constantino-garcia.r-universe.dev/nonlinearTseries), @garcia_nonlinearTseries_2026].

Yet, `pyunicorn` has maintained its unique position, bridging and complementing the above application areas while building on established graph packages. For instance, the backbone class `pyunicorn.Network` is built on top of the `python-igraph.Graph` class, extending the latter’s functionality to more advanced constructs, such as coupled multilayer networks [@donges_coupled_2011] or node-weighted (especially so-called node-splitting-invariant) network measures [@heitzig_nsi_2012; @wiedermann_interacting_nsi_2013; @zemp_dirweigh_nsi_2014]. Overall, as motivated in the previous section, the core strength of `pyunicorn` lies in its seamless integration of methods from complex network theory and nonlinear time series analysis. Thorough summaries of related concepts and algorithms can be found in @zou_networks_timeseries_2019, @silva_timeseries_networks_2021 and @marwan_palaeo_2021.

# Software design
\label{software_design}

The `pyunicorn` package follows an object-oriented design, and its class and module hierarchy separates the core data structures from the analysis methods using them. This allows for conceptual relationships between methods to be clearly reflected through inheritance. For example, the `RecurrenceNetwork` class inherits much of its functionality from the `Network` and `RecurrencePlot` classes (see fig. \ref{modules}) as recurrence networks per definition are a combination of both.

`Network` is the central class in `pyunicorn`, and is backed both by a `python-igraph` [@csardi_Igraph_2005] graph and by a `scipy.sparse` [@virtanen_scipy_2020] matrix. In addition to importing, constructing, querying, updating and exporting these internal data structures, `Network` is also responsible for implementing more specialized network annotations and graph algorithms, and for sampling from random graph models. Most other classes in the package derive from `Network`, extending it with dedicated logic for analyses. Many library methods are implemented by calling compiled functions, which are organized into `pyunicorn`’s own C/Cython [@behnel_cython_2011] extension modules, via array interfaces.

Overall, this software design covers many common use cases in complex system science: It supports data imports from a variety of formats and libraries, offers an API that is simple to navigate in terms of recognizable analysis methods, integrates naturally with the Python numerical computing ecosystem, and achieves a performance level for its graph algorithms that is comparable to, or competitive with, many other general-purpose implementations. `pyunicorn` is not specifically designed for out-of-core and for high-performance computing, but its extension modules provide a valuable starting point for developers interested in non-standard network measures.

![Overview of `pyunicorn`’s modules (e.g. `core`) and classes (e.g. `Network`), with an illustration of inheritance relations reflecting the conceptual relationships between methods, for the example of `RecurrenceNetwork`.\label{modules}](img/module_overview.png)

Since its first publication, the package has been updated multiple times to reflect changing standards in package management, extension compilation, linting, testing and continuous integration, and it now operates on Windows, macOS and Linux systems with all officially supported Python versions. Substantial additions to the test suite, as well as removal of experimental and untested code, have increased test coverage from less than 50% to slightly below 80%. The Python/Cython interface was overhauled, in order both to reflect changes in the Cython and Numpy [@harris_numpy_2020] toolchains and to protect against type- and size-related crashes, and most of the extension modules were ported from C to Cython in order to simplify maintenance.

The package has also grown by a range of new or updated functionality:

- Spatial and interacting network analyses were generalized by adding a `SpatialNetwork` class, by adding a Watts-Strogatz model to the `Network` class, and by adding new metrics to the `RecurrencePlot`, `CoupledClimateNetwork` and `InteractingNetwork` classes.
- Various metrics of the `Network` class, and especially their node-splitting-invariant versions, were generalized to support weighted and directed networks.
- A new `EventSeries` class was added in a dedicated module, accompanied by an `EventSeriesClimateNetwork` class extending the existing `climate` module.
- More recently, the `EventSeries` class was substantially revised to enable vectorisation, sparse computations and optional parallelization.
- A collection of tutorial `jupyter` notebooks was added to the documentation.
- The mix-in `Cached` consolidates the previously inconsistent mechanisms for object state management and for memoisation of expensive properties and methods.
- The `MapPlot` class was refactored and simplified.

A comprehensive record of additions since the original publication can be found in our [CHANGELOG](https://github.com/pik-copan/pyunicorn/blob/joss-paper/CHANGELOG.rst).

# Research impact statement
\label{research_impact_statement}

`pyunicorn` has been continuously employed in scientific research since its first release, and its multifaceted functionality has supported quantitative analyses in a wide range of disciplines:

- biology and medical research, e.g., statistical tests in cancer research [@amemiya_cancer_2025], EEG data analysis [@frolov_recurrence_2023; @lechner_depressive_2024], cortical networks during decision-making [@maksimenko_neural_2019], and mammal evolution [@bekeraite_evolutionary_2025];
- astrophysics, e.g., variability of active galactic nuclei [@phillipson_galactic_2023], black holes [@broadbent_cygnus_2023] and stars [@george_betelgeuse_2020];
- particle physics [@mullin_susy_2021];
- geophysics, e.g., crustal displacements [@hobbs_gnss_2018] and other applications [@talavera_mantle_2023; @donner_magnetosphere_2019];
- engineering, e.g., detection of thermoacoustic instabilities in rocket chambers [@waxenegger_rocket_2021] and complex response analysis in the dynamics of large machines [@geier_oscillators_2024];
- social science, e.g., detection of common features of societal marginalization [@schleussner_minorities_2016];
- climate and paleoclimate science [@sumit_india_2026; @jiang_tibet_2024; @moinat_tipping_2024; @haas_pitfalls_2023; @sun_texas_2018; @li_pakistan_2026; @marwan_palaeo_2021; @caesar_amoc_2020; @franke_anomalies_2017; @franke_holocene_2017; @wolf_baiu_2021; @wolf_itcz_2021; @wolf_connectivity_2021; @donges2015b; @nocke_visual_2015; @wiedermann_hierarchical_2017; @lekscha_windowed_2020], especially the study of El Niño related phenomena [@feng_enso_2017; @oluwole_enso_2017; @broni-bedaiko_enso_2019; @ekhtiari_enso_2021; @wiedermann_enso_2016];
- as well as methodological advances in network theory and nonlinear dynamics [@silini_entropy_2021; @yuan_correlations_2024; @medrano_radius_2021; @wiedermann_surrogates_2016; @odenweller_paired-event_2020; @sales_stickiness_2023; @subramaniyam_signatures_2015; @lekscha_phasespace_2018; @alberti_magnetic_2020].

In summary, `pyunicorn` has exhibited a notable uptake from the scientific community, with 26 out of the 46 surveyed publications originating from authors unaffiliated with the research groups of the package authors. Additionally, `pyunicorn`’s collection of tutorial notebooks provides valuable introductions to the relevant methodologies in nonlinear time series and complex network analysis, and has repeatedly been used as training material in academic teaching contexts.

# AI usage disclosure

No generative AI was used in the development of the software package as a whole, or in the writing of this manuscript. More recently, AI coding agents may have occasionally been used for maintenance or review tasks by `pyunicorn` contributors and maintainers.

# Author contributions

FK coordinated software maintenance and releases in recent years. BB contributed to software design and maintenance, provided technical guidance, and edited tutorials. MB contributed to software maintenance and wrote tutorials. FK and BB drafted the manuscript, and MB wrote the Research Impact Statement. MB, JFD and RVD reviewed and edited the manuscript. JFD and RVD organized funding and supervised development. FK and JFD coordinated the manuscript publication.

# Acknowledgements

We acknowledge all code and bug report contributions to `pyunicorn` since its open source release, see [CONTRIBUTIONS](https://github.com/pik-copan/pyunicorn/blob/joss-paper/CONTRIBUTIONS.rst). We are especially thankful for temporary support of the long-term development and maintenance work by Jonathan Kroenke, Nils Harmening, Johannes Kassel, Lena Schmidt, Ronja Hotz and Wolfram Barfuss, as well as a recent valuable contribution by Guruprem Bishnoi. The more recent development of `pyunicorn` was financially supported by the German Federal Ministry of Education and Research (BMBF) within the scope of the projects GOTHAM (grant no. 01LP1611A) and ROADMAP (grant no. 01LP2002B). We are grateful to `pyunicorn`'s original co-developer Jobst Heitzig for his availability for consultation. Lastly, FK would like to thank Jakob Harteg and Lorenz Sieben for inspiration on scientific software development in Python.

# References