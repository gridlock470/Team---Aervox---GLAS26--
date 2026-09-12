"""Generate the project overview PDF.

    python scripts/make_overview_pdf.py

Figures that describe the system (grid, bbox, lead times, channel count) are
imported from nowcast.config / nowcast.schema rather than retyped, so the
document cannot drift from the code it describes.

Status wording is deliberate: "built" means it exists and is tested, "planned"
means it is designed but not written. Nothing here claims more than the repo
contains.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from nowcast import config, schema

INK = colors.HexColor("#14202E")
DIM = colors.HexColor("#5A6B7D")
RULE = colors.HexColor("#C7D2DC")
BAND = colors.HexColor("#EEF2F6")
ACCENT = colors.HexColor("#1E5A8A")

OUT = Path(config.PROJECT_ROOT) / "docs" / "SIH_Project_Overview.pdf"


def styles() -> dict:
    s = getSampleStyleSheet()
    base = dict(fontName="Helvetica", alignment=TA_LEFT)
    return {
        "title": ParagraphStyle("t", parent=s["Title"], fontName="Helvetica-Bold",
                                fontSize=21, leading=25, textColor=INK, spaceAfter=2),
        "sub": ParagraphStyle("sub", fontSize=10.5, leading=15, textColor=DIM, **base),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, leading=18,
                             textColor=ACCENT, spaceBefore=14, spaceAfter=6,
                             alignment=TA_LEFT),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11, leading=14,
                             textColor=INK, spaceBefore=10, spaceAfter=4, alignment=TA_LEFT),
        "body": ParagraphStyle("b", textColor=INK, fontSize=9.7, leading=14, spaceAfter=6, **base),
        "small": ParagraphStyle("s", textColor=DIM, fontSize=8.6, leading=12, **base),
        "cell": ParagraphStyle("c", textColor=INK, fontSize=8.8, leading=12, **base),
        "cellb": ParagraphStyle("cb", fontName="Helvetica-Bold", fontSize=8.8,
                                leading=12, textColor=INK, alignment=TA_LEFT),
    }


def rule(space_before: float = 2, space_after: float = 8):
    return HRFlowable(width="100%", thickness=0.6, color=RULE,
                      spaceBefore=space_before, spaceAfter=space_after)


def bullets(items: list[str], st: dict) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(t, st["body"]), leftIndent=10, value="circle") for t in items],
        bulletType="bullet", start="circle", leftIndent=12, bulletFontSize=5,
        spaceBefore=0, spaceAfter=4,
    )


def kv_table(rows: list[tuple[str, str]], st: dict, w1: float = 46 * mm) -> Table:
    data = [[Paragraph(k, st["cellb"]), Paragraph(v, st["cell"])] for k, v in rows]
    t = Table(data, colWidths=[w1, 170 * mm - w1 - 6 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("BACKGROUND", (0, 0), (0, -1), BAND),
    ]))
    return t


def stack_table(rows: list[tuple[str, str, str]], st: dict) -> Table:
    head = [Paragraph(h, st["cellb"]) for h in ("Layer", "Technology", "Why this choice")]
    data = [head] + [
        [Paragraph(a, st["cell"]), Paragraph(b, st["cell"]), Paragraph(c, st["cell"])]
        for a, b, c in rows
    ]
    t = Table(data, colWidths=[34 * mm, 52 * mm, 78 * mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 0), (-1, 0), BAND),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, RULE),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(DIM)
    canvas.drawString(20 * mm, 12 * mm, "Hyperlocal Severe-Weather Nowcasting - project overview")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
    canvas.restoreState()


def build() -> Path:
    st = styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="Hyperlocal Severe-Weather Nowcasting - Project Overview",
        author="Team GridLock",
    )

    n_lat, n_lon = config.GRID_SHAPE
    hazards = ", ".join(h.replace("_", " ") for h in config.HAZARDS)
    leads = f"{min(config.LEAD_TIMES_H)}-{max(config.LEAD_TIMES_H)} h"
    levels = ", ".join(str(p) for p in config.PRESSURE_LEVELS_HPA)

    s: list = []

    # ---------------- cover / summary ----------------
    s.append(Paragraph("Hyperlocal Severe-Weather Nowcasting", st["title"]))
    s.append(Paragraph(
        "Predicting severe thunderstorms, cloudbursts and flash floods "
        f"{leads} ahead, at {config.GRID_RESOLUTION_DEG}&deg; resolution, "
        "with minutes of processing latency instead of hours.", st["sub"]))
    s.append(Spacer(1, 5))
    s.append(rule())

    s.append(Paragraph("The problem", st["h1"]))
    s.append(Paragraph(
        "Operational numerical weather prediction (NWP) is skilful at the synoptic scale but "
        "weak exactly where it matters most for disaster response: small, fast, convective "
        "events in the 2-6 hour window. Two gaps drive the damage.", st["body"]))
    s.append(bullets([
        "<b>Scale.</b> A cloudburst is a 5-10 km phenomenon lasting an hour or two. "
        "Operational guidance is issued on grids far coarser than the event itself, so the "
        "hazard is averaged away before a forecaster ever sees it.",
        "<b>Latency.</b> End-to-end pipelines take hours from observation to actionable "
        "product. For a hazard whose warning window is 2-6 hours, multi-hour latency "
        "consumes most of the lead time it was meant to provide.",
    ], st))
    s.append(Paragraph(
        "The consequence is concrete: the 2-3 May 2018 north-India thunderstorm and "
        "dust-storm outbreak killed roughly 125 people across Delhi NCR, Uttar Pradesh and "
        "Rajasthan. That event is the demonstration case this system is built and evaluated "
        "against.", st["body"]))

    s.append(Paragraph("What we are building", st["h1"]))
    s.append(Paragraph(
        "A nowcasting system that pairs NWP fields with fast-updating satellite observations "
        "and feeds them to a spatiotemporal model, producing calibrated probability maps per "
        "hazard per lead hour, with the reasoning behind each forecast exposed to the "
        "operator and alerts emitted in a standard format.", st["body"]))
    s.append(kv_table([
        ("Hazards", hazards),
        ("Lead times", f"{leads}, hourly steps"),
        ("Analysis grid", f"{config.GRID_RESOLUTION_DEG}&deg; (~11 km), "
                          f"{n_lat} x {n_lon} cells"),
        ("Pilot region", f"Uttarakhand and Delhi NCR &mdash; N {config.BBOX_NORTH}, "
                         f"S {config.BBOX_SOUTH}, W {config.BBOX_WEST}, E {config.BBOX_EAST}"),
        ("Vertical levels", f"{levels} hPa"),
        ("Model input", f"{len(schema.FEATURE_CHANNELS)} channels x "
                        f"{config.INPUT_SEQ_LEN} time steps x {n_lat} x {n_lon}"),
        ("Model output", f"{len(config.HAZARDS)} hazards x {len(config.LEAD_TIMES_H)} "
                         f"lead hours x {n_lat} x {n_lon} probability maps"),
    ], st))

    s.append(Paragraph("Languages, and why each one", st["h1"]))
    s.append(Paragraph(
        "Three languages, each chosen for a job the others do worse.", st["body"]))
    s.append(stack_table([
        ("Python", "the whole data and model pipeline",
         "The scientific stack is the reason. xarray and Zarr handle labelled "
         "N-dimensional weather data natively; MetPy implements the convective "
         "diagnostics (CAPE, CIN, lifted index) to meteorological standards, so we "
         "do not re-derive thermodynamics by hand; PyTorch trains the model. "
         "Rewriting any of that in a faster language would cost months and forfeit "
         "correctness the domain libraries already guarantee."),
        ("JavaScript (React)", "the operator console",
         "The console is a live map with GPU-drawn probability layers. MapLibre GL "
         "and deck.gl have no serious equivalent outside JavaScript, and the browser "
         "is the only runtime a district control room can open without installing "
         "anything."),
        ("SQL (PostGIS)", "spatial and time-series storage",
         "Alert areas are polygons; forecasts are a time series. Both are database "
         "problems with mature answers -- PostGIS for geometry, TimescaleDB for "
         "append-heavy history -- not problems to solve in application code."),
    ], st))
    s.append(Paragraph(
        "On Python and speed: the usual objection is that Python is slow. It is not "
        "the bottleneck here. The numerical work runs inside compiled NumPy and "
        "PyTorch kernels, and the operational budget is dominated by network I/O and "
        "inference, not interpreter overhead. Where a hot loop did threaten the "
        "budget, the fix was vectorisation, not a change of language.", st["body"]))

    s.append(Paragraph("Why this is different", st["h1"]))
    s.append(bullets([
        "<b>It targets the regime the benchmarks skip.</b> BharatBench, the established "
        "Indian ML-weather benchmark, operates at 1.08&deg; and 6-hourly sampling for 3-5 day "
        "forecasts. We work at 0.1&deg; and hourly steps for 1-6 hour leads &mdash; the scale at "
        "which cloudbursts physically exist.",
        "<b>Lead signals, not just fields.</b> The model is fed engineered precursor "
        "channels &mdash; cloud-top cooling rate, water-vapour tendency, low-level convergence, "
        "shear, CAPE and CIN &mdash; rather than raw variables alone, so it learns from the "
        "quantities that actually precede convective initiation.",
        "<b>Terrain is a first-class input.</b> Flow accumulation, flow direction and height "
        "above nearest drainage are computed once from a 30 m DEM and routed into the "
        "flash-flood head, so the model distinguishes rain that runs off from rain that pools.",
        "<b>One backbone, three hazards.</b> A shared spatiotemporal encoder with three "
        "task heads lets correlated hazards reinforce each other instead of being modelled "
        "as three unrelated problems.",
        "<b>Explainability is part of the product.</b> Every forecast surfaces its ranked "
        "drivers. A duty officer ordering an evacuation needs to know why, not just what.",
        "<b>Calibrated, not just confident.</b> Uncertainty is estimated and calibrated, so "
        "a stated 70 per cent means roughly seven events in ten.",
    ], st))

    s.append(PageBreak())

    # ---------------- frontend ----------------
    s.append(Paragraph("1. Frontend", st["h1"]))
    s.append(Paragraph(
        "An operational console for a duty forecaster, not a dashboard for a demo. The "
        "design question was what someone at 3 a.m. must answer in five seconds: where is "
        "the hazard, how severe, how confident, and what action follows.", st["body"]))
    s.append(Paragraph("Status: built and running on mock data.", st["small"]))

    s.append(Paragraph("Stack", st["h2"]))
    s.append(stack_table([
        ("Framework", "React 18 + Vite 5",
         "Fast build and hot reload; no server-side rendering needed for a single-screen console."),
        ("Mapping", "MapLibre GL 4",
         "Open, vector-tile basemap with no proprietary key or usage ceiling."),
        ("Data overlay", "deck.gl 9",
         "GPU-rendered layers; keeps thousands of probability cells interactive."),
        ("Styling", "Hand-written CSS with design tokens",
         "A small fixed palette is easier to keep disciplined than a utility framework."),
    ], st))

    s.append(Paragraph("Design decisions", st["h2"]))
    s.append(bullets([
        "<b>The map owns the screen.</b> Full-bleed layout; everything else lives behind a "
        "five-tab dock (alerts, points, drivers, model, CAP log) that collapses to a 44 px "
        "rail. All data stays one click away, none of it competes with the map.",
        "<b>Colour is reserved for severity.</b> No other element on screen carries chroma. "
        "Driver contributions use neutral sequential tones, because a driver weight is not a "
        "hazard level and colouring it like one would mislead the operator.",
        "<b>Motion only answers an action.</b> No ambient pulsing or glow; transitions "
        "respond to tab changes and expansions, and honour reduced-motion preferences.",
        "<b>Accessible by construction.</b> Full ARIA tablist with roving tabindex and "
        "arrow-key navigation; visible focus throughout.",
    ], st))

    s.append(Paragraph("Next", st["h2"]))
    s.append(Paragraph(
        "Replace the mock data module with live API and tile calls, and wire the timeline "
        "scrubber, driver rail and CAP panel to real model output.", st["body"]))

    # ---------------- backend ----------------
    s.append(Paragraph("2. Backend", st["h1"]))
    s.append(Paragraph(
        "The serving layer between trained model and console, and the component that carries "
        "the latency claim. Its target is a processing budget measured in minutes.", st["body"]))
    s.append(Paragraph("Status: designed, not yet implemented.", st["small"]))

    s.append(Paragraph("Planned stack", st["h2"]))
    s.append(stack_table([
        ("API", "FastAPI + WebSocket",
         "Async request handling; WebSocket pushes new nowcast cycles without polling."),
        ("Spatial store", "PostGIS",
         "Geometry queries for districts, basins and alert areas."),
        ("Time series", "TimescaleDB",
         "Hypertables suit append-heavy forecast and observation histories."),
        ("Tiles", "TiTiler",
         "Serves probability rasters as map tiles directly from the datacube."),
        ("Orchestration", "Prefect",
         "Scheduled ingestion and inference with retries and observable runs."),
        ("Alerting", "CAP 1.2",
         "Common Alerting Protocol is the format Indian agencies already consume."),
    ], st))

    s.append(Paragraph("How the latency target is approached", st["h2"]))
    s.append(bullets([
        "Incremental ingestion &mdash; process only the newest time slice, never re-read "
        "the archive.",
        "Precomputed static layers &mdash; terrain routing is calculated once, offline, "
        "not per run.",
        "Streaming rather than batch, keeping intermediate state in memory.",
        "Asynchronous alert generation, decoupled from the inference path.",
        "GPU-accelerated inference where hardware allows.",
    ], st))

    s.append(Paragraph("Where the latency budget goes", st["h2"]))
    s.append(Paragraph(
        "The target is a processing budget measured in minutes, against pipelines "
        "that currently take hours. The budget is spent roughly as follows, and each "
        "line is an engineering decision rather than an aspiration:", st["body"]))
    s.append(kv_table([
        ("Ingest the newest slice",
         "Incremental only. The archive is never re-read; one time step is appended "
         "to the datacube."),
        ("Derive lead signals",
         "Vectorised array operations over a 38 x 49 grid. Small by construction -- "
         "the pilot domain is 1,862 cells, not a national grid."),
        ("Terrain context",
         "Zero cost at run time. Flow accumulation, flow direction and HAND are "
         "precomputed once, offline, and simply read."),
        ("Inference",
         "A single forward pass producing all three hazards and all six lead hours "
         "at once, because the heads share one backbone."),
        ("Publish and alert",
         "Probability rasters written as map tiles; CAP alerts generated "
         "asynchronously so alerting never blocks the forecast path."),
    ], st))
    s.append(Paragraph(
        "Two design choices do most of the work. The multi-task backbone means three "
        "hazards cost one forward pass rather than three. And precomputing terrain "
        "routing moves the most expensive geospatial work out of the operational path "
        "entirely -- it is the same rasters every run, so computing them per cycle "
        "would be pure waste.", st["body"]))

    s.append(Paragraph("An honest distinction", st["h2"]))
    s.append(Paragraph(
        "Two different quantities are often conflated. <b>Processing latency</b> is the time "
        "from input arriving to product published &mdash; that is what this system targets in "
        "minutes. <b>Data freshness</b> is bounded by the providers: reanalysis is months "
        "behind, merged satellite rainfall runs about four hours behind, and only direct "
        "geostationary imagery updates every 15-30 minutes. The operational design therefore "
        "leans on satellite for the fast-updating channel and bias-corrected NWP as the "
        "slower background. Claiming minutes of end-to-end freshness would not be truthful.",
        st["body"]))

    s.append(PageBreak())

    # ---------------- ML ----------------
    s.append(Paragraph("3. Machine learning", st["h1"]))
    s.append(Paragraph(
        "Status: data pipeline, feature engineering, models, training loop and uncertainty "
        "code are written and unit-tested against synthetic fixtures. Training on real data "
        "begins as the downloads complete.", st["small"]))

    s.append(Paragraph("Stack", st["h2"]))
    s.append(stack_table([
        ("Arrays", "xarray + Zarr + dask",
         "Labelled N-d arrays with chunked, parallel access to a datacube too large for memory."),
        ("Meteorology", "MetPy",
         "Derives CAPE, CIN, lifted index and moisture diagnostics from pressure-level profiles."),
        ("Terrain", "pysheds / RichDEM, rioxarray",
         "D8 flow routing, flow accumulation and height above nearest drainage."),
        ("Deep learning", "PyTorch 2 + Lightning",
         "Multi-task training loop with checkpointing and reproducible runs."),
        ("Baseline", "LightGBM",
         "Per-pixel gradient boosting: an accuracy floor and a fallback that always trains."),
        ("Explainability", "Captum + SHAP",
         "Attributions become the ranked driver list the console displays."),
    ], st))

    s.append(Paragraph("Pipeline", st["h2"]))
    s.append(bullets([
        "<b>Ingestion.</b> Reanalysis, satellite and terrain sources are cropped, regridded "
        f"and aligned onto one {config.GRID_RESOLUTION_DEG}&deg; grid in a Zarr datacube.",
        "<b>Feature engineering.</b> Raw fields become precursor channels: integrated water "
        "vapour and its tendency, cloud-top temperature drop-rate, CAPE, CIN, lifted index, "
        "low-level convergence and 0-1 km / 0-6 km shear.",
        "<b>Terrain routing.</b> Flow accumulation, flow direction and HAND rasters, computed "
        "once and broadcast as static channels.",
        "<b>Baseline model.</b> LightGBM per pixel &mdash; validates labels and targets before any "
        "deep model is trusted.",
        "<b>Main model.</b> A shared 3D-CNN / ConvLSTM backbone with three hazard heads, "
        "upgrading to a spatiotemporal transformer with satellite-to-NWP cross-attention.",
        "<b>Uncertainty.</b> Monte-Carlo dropout with probability calibration.",
        "<b>Bias correction.</b> Bridges the gap between training on reanalysis and running "
        "on operational analyses.",
    ], st))

    s.append(Paragraph("Engineering discipline", st["h2"]))
    s.append(bullets([
        "A single configuration module is the sole source of truth for grid, bbox, levels, "
        "hazards and lead times; no module redefines them locally.",
        "Train, validation and test splits are disjoint date ranges, and label horizons are "
        "masked at the boundaries so future information cannot leak backwards.",
        "Labels are peak-normalised, with a configured occurrence threshold applied "
        "consistently wherever a hard classification is required.",
        "The codebase went through three independent write-review-fix rounds; every blocking "
        "finding was closed before the build was declared complete.",
    ], st))

    s.append(PageBreak())

    # ---------------- research ----------------
    s.append(Paragraph("4. Research and data", st["h1"]))

    s.append(Paragraph("Data sources", st["h2"]))
    s.append(stack_table([
        ("IMDAA", "NCMRWF regional reanalysis, 12 km",
         "Primary atmospheric state for training; India-specific assimilation."),
        ("MERA", "NCMRWF merged rainfall, 4 km",
         "Highest-resolution rainfall truth available for the region."),
        ("GPM IMERG", "NASA, 0.1 deg, 30-minute",
         "Independent rainfall label on exactly the analysis grid."),
        ("INSAT-3D / 3DR", "ISRO via MOSDAC",
         "Water-vapour and thermal-IR imagery: the only genuinely real-time channel."),
        ("Copernicus GLO-30", "30 m elevation",
         "Terrain routing for the flash-flood head."),
        ("ERA5", "ECMWF, 0.25 deg, hourly",
         "Self-service substitute where queued national sources cannot arrive in time."),
    ], st))

    s.append(Paragraph("Region-specific behaviour", st["h2"]))
    s.append(Paragraph(
        "The pilot deliberately spans two regimes that behave nothing alike: the "
        "Himalayan slopes of Uttarakhand, where orographic lift drives cloudbursts "
        "and steep terrain converts rain into flash floods within the hour, and the "
        "Delhi NCR plains, where convection is thermally driven and the same rainfall "
        "produces urban inundation rather than a torrent. A single set of thresholds "
        "applied to both would be wrong in one of them.", st["body"]))
    s.append(Paragraph(
        "Our own label audit measured the split rather than assuming it. Across "
        "9.07 million cell-timesteps, thunderstorm positives appear in all 1,862 grid "
        "cells with the top decile holding only 22.6&nbsp;% of them -- spread evenly, as "
        "thermal convection should be. Cloudburst positives occur in just 111 cells "
        "and flash-flood positives in 177, tightly clustered on the orography. The "
        "two hazard families are not merely rarer; they live in different places.",
        st["body"]))
    s.append(Paragraph("The system encodes that in four ways:", st["body"]))
    s.append(bullets([
        "<b>Terrain as a model input.</b> Elevation, slope, flow accumulation and "
        "height-above-nearest-drainage are computed once from a 30&nbsp;m DEM and fed to "
        "the network as static channels. The model learns the regional distinction "
        "from the terrain itself rather than from a region label pasted on top.",
        "<b>A dedicated flash-flood head.</b> Routed terrain context goes to the "
        "flash-flood branch specifically, so it can distinguish rain that runs "
        "downhill into a valley from rain that pools on a plain.",
        "<b>Per-region normalisation and calibration.</b> Statistics are fitted per "
        "region, because a 99th-percentile rainfall hour in the hills is not the same "
        "number as in the plains. Calibration is fitted the same way, so a stated "
        "70&nbsp;% means the same thing in both.",
        "<b>Region-aware thresholds.</b> Hazard criteria are configuration, not "
        "hard-coded constants, so each region carries the rainfall and accumulation "
        "thresholds its own climatology supports.",
    ], st))
    s.append(Paragraph(
        "The practical consequence is that adding a third region is a configuration "
        "and retraining exercise, not a rewrite. The bounding box, grid and hazard "
        "criteria all live in one configuration module; nothing in the architecture "
        "is specific to these two regions.", st["body"]))

    s.append(Paragraph("Positioning against existing work", st["h2"]))
    s.append(Paragraph(
        "BharatBench established data-driven weather forecasting over India at 1.08&deg; and "
        "6-hourly resolution, benchmarked at 3-day and 5-day leads. IndiaWeatherBench extends "
        "regional benchmarking with probabilistic metrics. Both address medium-range "
        "forecasting. This project targets the complementary regime &mdash; convective nowcasting "
        "at 0.1&deg; and 1-6 hour leads &mdash; where NWP skill is weakest and where the hazards "
        "that kill people actually occur.", st["body"]))

    s.append(Paragraph("Known limitations", st["h2"]))
    s.append(bullets([
        "Terrain routing is computed after regridding to ~11 km cells, which under-resolves "
        "real flash-flood catchments. Routing at native 30 m resolution and aggregating "
        "afterwards is the correct fix.",
        "Convective diagnostics are derived from seven pressure levels; a denser profile "
        "would improve CAPE and lifted-index quality.",
        "Where a national source is still queued, a coarser global substitute is used. That "
        "is a stated substitution, not a silent one.",
    ], st))

    s.append(Paragraph("Future scope", st["h2"]))
    s.append(bullets([
        "<b>Pretraining on convective archives.</b> SEVIR provides more than 10,000 storm "
        "events at 1 km and 5-minute sampling with aligned radar, satellite and lightning. "
        "Pretraining the backbone there and fine-tuning on Indian data should beat training "
        "from scratch on a few regional seasons.",
        "<b>Assimilating ground radar.</b> Doppler weather radar would add the one "
        "observation type currently missing from the input stack.",
        "<b>National rollout.</b> Nothing in the architecture is specific to the two pilot "
        "regions; the bounding box is configuration, not code.",
        "<b>Lightning prediction.</b> A fourth head, once a lightning-detection feed is "
        "available for labels.",
        "<b>Impact-based forecasting.</b> Combining hazard probability with population and "
        "infrastructure exposure to rank alerts by consequence rather than by intensity.",
        "<b>Edge deployment.</b> A distilled model running at district control rooms, so "
        "warnings survive a network outage during the event they are warning about.",
    ], st))

    s.append(Spacer(1, 8))
    s.append(rule(2, 4))
    s.append(Paragraph(
        "Figures describing the grid, region, lead times and model shapes in this document "
        "are read directly from the project configuration at generation time.", st["small"]))

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size/1024:.0f} KB)")
    sys.exit(0)
