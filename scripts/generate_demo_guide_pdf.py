#!/usr/bin/env python3
"""Generates PATHSNAP_DEMO_GUIDE.pdf with ReportLab.
A comprehensive, publication-quality presentation and defense guide.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "PATHSNAP_DEMO_GUIDE.pdf"


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        if self._pageNumber > 1:
            self.drawString(54, 755, "PathSnap: Live Demo & Technical Defense Guide")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 750, 558, 750)
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "PES University | Map-Matching & Route Reconstruction Demo Guide")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    NAVY = colors.HexColor("#1e3a8a")
    SLATE = colors.HexColor("#334155")
    BLUE_BG = colors.HexColor("#f0f7ff")
    BORDER_COL = colors.HexColor("#cbd5e1")
    HEADER_BG = colors.HexColor("#1e3a8a")
    ACCENT_GREEN = colors.HexColor("#15803d")
    ACCENT_RED = colors.HexColor("#b91c1c")

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=NAVY, spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSub', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10, leading=14, textColor=colors.HexColor("#475569"), spaceAfter=10
    )
    h1_style = ParagraphStyle(
        'SectionH1', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=NAVY, spaceBefore=12, spaceAfter=5, keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'SectionH2', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=9.5, leading=12, textColor=colors.HexColor("#0f172a"), spaceBefore=7, spaceAfter=3, keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyDark', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8, leading=11, textColor=SLATE, spaceAfter=4
    )
    callout_style = ParagraphStyle(
        'Callout', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=8, leading=11, textColor=colors.HexColor("#1e293b")
    )
    table_cell = ParagraphStyle(
        'TableCell', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=SLATE
    )
    table_header = ParagraphStyle(
        'TableH', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=colors.white
    )

    def build_table(header_data, rows_data, col_widths=None):
        data = [[Paragraph(c, table_header) for c in header_data]]
        for row in rows_data:
            data.append([Paragraph(str(c), table_cell) for c in row])
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COL),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        return t

    story = []

    # Title Banner
    story.append(Paragraph("PATHSNAP: DEMO GUIDE & TECHNICAL CHEAT-SHEET", title_style))
    story.append(Paragraph("<b>Two-Stage Map-Matching (XGBoost + Bounded Network HMM)</b> | Presentation Script, Math Breakdown & Defense Q&A", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=8))

    # Meta Quick Reference Box
    quick_ref = [
        ["Demo URL (Laptop 2)", "http://10.1.4.66:8080/?api=http://10.1.4.66:8000", "Backend (Laptop 1)", "http://10.1.4.66:8000 (FastAPI)"],
        ["Evaluator-Test Point Acc", "95.82% (HMM) vs 91.91% (Classifier)", "Route F1 Score", "98.44% (Recall: 99.61%, Precision: 97.35%)"],
        ["Route Mismatch Reduction", "3.24% (dropped from 30.68% - 9.5x drop)", "Average Latency", "176.2 ms / trip (Real-Time Sub-200ms)"],
        ["Public Repository", "github.com/ack-chai/pathsnap", "Core Model", "XGBoost (Bayesian-Optimised, 30 Optuna trials)"]
    ]
    box_data = [[Paragraph(f"<b>{r[0]}:</b> {r[1]}", table_cell), Paragraph(f"<b>{r[2]}:</b> {r[3]}", table_cell)] for r in quick_ref]
    t_box = Table(box_data, colWidths=[250, 254])
    t_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BLUE_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COL),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_box)
    story.append(Spacer(1, 8))

    # ------------------ SECTION 1 ------------------
    story.append(Paragraph("1. System Architecture: The Decoupled Two-Stage Approach", h1_style))
    story.append(Paragraph(
        "<b>The Problem:</b> Raw GPS fixes exhibit 5–20 m spatial error, sensor jitter, multipath urban canyon reflections, and sampling gaps. A naive classifier matching each GPS point independently to the nearest street causes severe zigzagging between parallel roads and one-way streets.<br/>"
        "<b>The Solution:</b> PathSnap decouples <i>local geometric/kinematic scoring</i> from <i>global topological graph connectivity</i>:",
        body_style
    ))

    arch_rows = [
        ["Stage 0: Metric Projection", "Converts spherical (lon, lat) to planar (x, y) meters via local equirectangular projection centered at trip midpoint. Enables fast Euclidean 2D distance calculations with <0.01% distortion."],
        ["Stage 1: Candidate Search", "A spatial KD-Tree extracts all road arcs within radius R = 60m (enforcing 3 to 12 candidates). Candidate miss rate is only 0.09% (true arc is present in 99.91% of candidate sets)."],
        ["Stage 1b: 23 Features", "Computes 23 engineered features across 5 groups: Proximity (dist, dist_rank, gaps), Window smoothing (win_dist across 5 points), Kinematics (speed, heading_diff, heading_cos), Topology (degrees, one-way/two-way), and Jumps."],
        ["Stage 1c: XGBoost Classifier", "Evaluates all candidate pairs to produce calibrated posterior probabilities p(y=1 | z_t, a_i). Trained via 5-fold GroupKFold cross-validation (grouped by trip ID). Sub-millisecond inference (0.84 ms/1k rows)."],
        ["Stage 2: Bounded HMM", "Hidden states = candidate arcs. Emission = temperature-scaled classifier probability. Transition = exponential penalty on difference between network shortest-path driving distance (bounded Dijkstra) and straight-line GPS displacement."],
        ["Stage 2b: Viterbi Trellis", "Dynamic programming finds the globally optimal path across all time steps. Automatically splits trajectories at time gaps > 180s. Stitches intermediate arcs for continuous polylines."]
    ]
    story.append(build_table(["Pipeline Stage", "Technical Function"], arch_rows, [110, 394]))
    story.append(Spacer(1, 8))

    # ------------------ SECTION 2 ------------------
    story.append(Paragraph("2. Mathematical Foundations (Under the Hood)", h1_style))
    story.append(Paragraph(
        "<b>1. Local Equirectangular Metric Projection:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;x = R_earth · (lon - lon_0) · cos(lat_0) · (π / 180), &nbsp;&nbsp;&nbsp;&nbsp; y = R_earth · (lat - lat_0) · (π / 180)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;Where R_earth = 6,371,000 m. Replaces expensive spherical Haversine formulas with fast Euclidean dot products.<br/>"
        "<b>2. Heading Disambiguation for Two-Way Streets:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;heading_diff = |θ_GPS - θ_arc| ∈ [0, π], &nbsp;&nbsp;&nbsp;&nbsp; heading_cos = cos(θ_GPS - θ_arc)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;When driving forward on an arc, heading_cos ≈ +1; for the reverse arc on a two-way street, heading_cos ≈ -1.<br/>"
        "<b>3. HMM Emission Log-Probability (Temperature Power γ = 0.5):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;ln P(z_t | a_i) = γ · ln( max(p_clf(a_i), 10^-6) )<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;γ = 0.5 softens overconfident classifier probabilities (e.g. 0.999), ensuring network transition penalties can override local geometric ambiguities.<br/>"
        "<b>4. HMM Transition Log-Probability (Scale Parameter β = 5.0 m):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;ln P(a_j^(t) | a_i^(t-1)) = - |d_network(a_i, a_j) - d_Euclidean(z_t-1, z_t)| / β<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;d_network is the exact driving distance computed by bounded Dijkstra search on the directed graph. If unreachable within physical speed limits (50 m/s), a penalty of -30.0 is applied.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # ------------------ SECTION 3 ------------------
    story.append(PageBreak())
    story.append(Paragraph("3. Master Performance Scorecard (All Partitions)", h1_style))
    
    perf_headers = ["Partition", "Strategy", "Point Accuracy", "Route Continuity", "Route Recall", "Route Precision", "Route F1", "Route Mismatch"]
    perf_rows = [
        ["Train Set (80 trips)", "Classifier Only", "89.44%", "94.95%", "—", "—", "83.24%", "48.41%"],
        ["Train Set (80 trips)", "Classifier + HMM", "93.53%", "97.31%", "98.98%", "95.07%", "96.77%", "7.37%"],
        ["Student-Test (10 trips)", "Classifier Only", "90.53%", "95.60%", "—", "—", "86.37%", "32.85%"],
        ["Student-Test (10 trips)", "Classifier + HMM", "94.73%", "97.93%", "99.66%", "98.09%", "98.86%", "2.31%"],
        ["Evaluator-Test (10 trips)", "Classifier Only", "91.91%", "94.49%", "—", "—", "86.82%", "30.68%"],
        ["Evaluator-Test (10 trips)", "Classifier + HMM", "95.82%", "96.50%", "99.61%", "97.35%", "98.44%", "3.24%"]
    ]
    story.append(build_table(perf_headers, perf_rows, [95, 80, 55, 52, 54, 56, 54, 58]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Evaluator-Test Trip-by-Trip Accuracy Breakdown (10 Trips):", h2_style))
    trip_headers = ["Trip ID", "GPS Points", "Classifier Acc", "HMM Acc", "Accuracy Gain", "Continuity", "Route F1", "Mismatch", "Latency"]
    trip_rows = [
        ["00000015", "6,854", "70.63%", "93.33%", "+22.70 pp", "99.82%", "93.06%", "14.78%", "436 ms"],
        ["00000017", "1,346", "93.76%", "95.17%", "+1.41 pp", "87.96%", "98.63%", "2.77%", "216 ms"],
        ["00000023", "458", "94.32%", "95.63%", "+1.31 pp", "98.47%", "98.72%", "2.57%", "42 ms"],
        ["00000030", "654", "82.72%", "86.85%", "+4.13 pp", "91.27%", "97.50%", "5.08%", "78 ms"],
        ["00000049", "2,642", "99.17%", "100.00%", "+0.83 pp", "99.43%", "99.94%", "0.12%", "230 ms"],
        ["00000059", "1,031", "96.61%", "97.67%", "+1.07 pp", "92.82%", "99.55%", "0.91%", "116 ms"],
        ["00000067", "916", "95.09%", "96.18%", "+1.09 pp", "99.13%", "97.86%", "4.35%", "61 ms"],
        ["00000073", "1,144", "98.08%", "98.51%", "+0.44 pp", "99.13%", "99.93%", "0.14%", "107 ms"],
        ["00000087", "1,399", "95.78%", "99.71%", "+3.93 pp", "99.07%", "99.77%", "0.46%", "194 ms"],
        ["00000094", "3,230", "92.94%", "95.17%", "+2.23 pp", "97.86%", "99.42%", "1.16%", "283 ms"],
        ["Average", "1,967", "91.91%", "95.82%", "+3.91 pp", "96.50%", "98.44%", "3.24%", "176 ms"]
    ]
    story.append(build_table(trip_headers, trip_rows, [54, 48, 55, 50, 58, 48, 48, 45, 48]))
    story.append(Spacer(1, 8))

    # ------------------ SECTION 4 ------------------
    story.append(Paragraph("4. Step-by-Step Live Demo Presentation Script (2 Minutes)", h1_style))
    
    script_steps = [
        ("0:00 - 0:30", "Introduction & Problem Statement",
         "\"Hello! This is PathSnap, our map-matching and route reconstruction system. In urban areas, GPS noise, tall buildings, and parallel roads make purely geometric matching fail. We solve this with a two-stage hybrid design: an XGBoost classifier that evaluates 23 domain features, followed by a Hidden Markov Model with Viterbi decoding that enforces physical road network connectivity.\""),
        ("0:30 - 0:50", "The Distributed Architecture",
         "\"Our system runs distributed across two laptops. Laptop 1 runs our FastAPI backend serving the model and holding ground-truth routes securely server-side. Laptop 2 runs this lightweight Leaflet web UI over local Wi-Fi. It queries POST /predict and renders the reconstructed route and live scorecards in real time (average latency: 176 ms).\""),
        ("0:50 - 1:20", "Click Trip 00000049 (Flawless Performance)",
         "\"Let's click trip 00000049 (2,642 GPS points). In 230 milliseconds, it achieves 100.00% Point Accuracy and 99.94% Route F1. Blue dots show the raw noisy GPS fixes; the green line shows our continuous predicted route accurately hugging the road network without cutting corners.\""),
        ("1:20 - 1:50", "Click Trip 00000015 (The Hero Demonstration)",
         "\"Now look at the hardest trip: 00000015 (6,854 points). Notice the Classifier-Only accuracy was only 70.63% because of severe GPS multipath reflections across parallel streets. But the HMM boosted accuracy to 93.33% (+22.70 pp). In the layer controls, if we toggle the orange dashed line (classifier-only), you see lateral jitter; the HMM's network transition penalty eliminated all zigzagging.\""),
        ("1:50 - 2:00", "Conclusion & Deliverables",
         "\"Across all 10 evaluator trips, Point Accuracy is 95.82%, Route Recall is 99.61%, and Route F1 is 98.44%. The complete code, trained models, interactive Folium maps, and our 9-page technical report are pushed and public on GitHub at github.com/ack-chai/pathsnap. Thank you!\"")
    ]
    for timing, step_title, spoken_words in script_steps:
        story.append(Paragraph(f"<b>[{timing}] {step_title}</b>", h2_style))
        card_table = Table([[Paragraph(spoken_words, callout_style)]], colWidths=[504])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COL),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(card_table)
        story.append(Spacer(1, 4))

    # ------------------ SECTION 5 ------------------
    story.append(PageBreak())
    story.append(Paragraph("5. Evaluator Defense Q&A Cheat-Sheet (10 Technical Questions)", h1_style))

    qa_list = [
        ("Q1: Why did you choose XGBoost over Random Forest or Deep Learning?",
         "We evaluated 10 models across 6 families using 5-fold GroupKFold cross-validation. While Random Forest had slightly higher CV Top-1 (87.74% vs 87.64%), XGBoost achieved the highest holdout Top-1 (89.83%), lowest log loss (0.0632), and 2.6x faster inference (0.84 ms/1k rows). Well-calibrated log loss is vital because classifier probabilities directly feed the HMM as emission scores. Deep learning (MLP) suffered from tabular feature scaling sensitivity and scored lower (88.74%)."),
        
        ("Q2: How did you prevent data leakage during training?",
         "We grouped strictly by `trip_id` using GroupKFold across the 80 training trips. All candidate rows from the same trip remained together in either train or validation, preventing geographic memorization. Feature transformers were fitted strictly on training folds. Neither student-test nor evaluator-test was ever seen during training or tuning."),
        
        ("Q3: How does the HMM transition probability work?",
         "It uses an exponential penalty on the difference between network shortest-path distance and Euclidean displacement: ln P(a_j | a_i) = -|d_network - d_Euclidean| / β, where β = 5.0 m. d_network is computed via bounded Dijkstra search on the directed road graph. If driving straight on a road, network distance equals Euclidean distance (penalty = 0). If an arc requires an impossible detour or u-turn, the discrepancy is large, heavily penalizing that transition."),
        
        ("Q4: Why is emission power temperature γ = 0.5 necessary?",
         "Tree ensembles can output extreme probabilities (e.g. 0.999), which in standard HMMs would overpower the transition penalty and re-introduce zigzagging. Scaling emission log-probabilities by γ = 0.5 softens the distribution, ensuring network topology and local geometry have balanced influence."),
        
        ("Q5: How do you handle two-way streets with opposing directions?",
         "A two-way street has two separate directed arcs. We engineered `heading_diff` (|θ_GPS - θ_arc|) and `heading_cos` (cos(Δθ)). Driving forward yields cos(Δθ) ≈ +1; driving against traffic yields cos(Δθ) ≈ -1. This allows the classifier to cleanly eliminate reverse candidates even when orthogonal distance is identical."),
        
        ("Q6: What happens during GPS tunnels, parking, or large signal gaps?",
         "If consecutive GPS fixes exhibit a time gap Δt > 180 seconds, PathSnap automatically splits the trajectory, resetting the Viterbi trellis. This prevents the Dijkstra search from finding absurd shortest paths across the city during stops."),
        
        ("Q7: How are reference routes aligned to GPS fixes for training labels?",
         "Because .route files provide an ordered list of arcs without timestamps, a monotone dynamic programming (DP) algorithm aligns each GPS point to an arc in the reference route by minimizing orthogonal distance while preserving sequence order. Points > 100 m from their aligned arc are marked unreliable and excluded from training."),
        
        ("Q8: Why local equirectangular metric projection instead of Haversine or UTM?",
         "Haversine requires repeated expensive trigonometric calls (sin, cos, atan2). UTM has zone boundary discontinuities. Equirectangular metric projection centered at the trip centroid creates a continuous planar coordinate system in meters with <0.01% distortion over city scales (<30 km), accelerating point-to-segment distance computations by over 20x."),
        
        ("Q9: What is candidate miss rate and how did you measure it?",
         "Candidate miss rate is the percentage of GPS fixes where the true road arc was not captured within the 60 m radius KD-tree search. In our evaluation dataset, candidate miss rate was only 0.09% (hit rate of 99.91%), proving the 60 m spatial window is both comprehensive and computationally tight."),
        
        ("Q10: How does the system ensure zero label leakage during live prediction?",
         "PathSnap.predict() reads strictly three files: .track, .nodes, and .arcs. The .route file is never opened or loaded during feature extraction, candidate classification, or HMM decoding. It is accessed solely by PathSnap.evaluate() post-prediction for scoring.")
    ]

    for q, a in qa_list:
        story.append(Paragraph(f"<b>{q}</b>", h2_style))
        story.append(Paragraph(a, body_style))
        story.append(Spacer(1, 2))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated Demo Guide PDF: {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build_pdf()

