"""
report_gen.py
--------------
Generates the downloadable PDF claim report required by the hackathon rubric.
Includes: claim details, original photo, XAI heatmap, AI prediction +
confidence, LLM summary, and the adjuster's final Human-in-the-Loop decision.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

BASE = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE, "reports")


def generate_report(claim: dict) -> str:
    """
    claim dict expects:
      claim_id, description, vehicle_age, claimed_amount, policy_type,
      predicted_class, confidence, all_confidences, summary, suggested_payout,
      original_image_path, heatmap_image_path,
      adjuster_decision ('approved'/'rejected'/'edited'), adjuster_notes,
      final_payout
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f"claim_{claim['claim_id']}_report.pdf")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18)
    h2 = styles["Heading2"]
    body = styles["BodyText"]

    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = []

    story.append(Paragraph("AI Co-Pilot: Insurance Claim Report", title_style))
    story.append(Spacer(1, 12))

    meta_table = Table([
        ["Claim ID", str(claim["claim_id"])],
        ["Policy Type", claim["policy_type"]],
        ["Vehicle Age", f"{claim['vehicle_age']} years"],
        ["Claimed Amount", f"${claim['claimed_amount']:,.2f}"],
    ], colWidths=[150, 300])
    meta_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Claimant Description", h2))
    story.append(Paragraph(claim["description"], body))
    story.append(Spacer(1, 12))

    # Images side by side
    if os.path.exists(claim["original_image_path"]) and os.path.exists(claim["heatmap_image_path"]):
        img_table = Table([
            [RLImage(claim["original_image_path"], width=180, height=180),
             RLImage(claim["heatmap_image_path"], width=180, height=180)],
            [Paragraph("Original Photo", body), Paragraph("XAI Heatmap (region importance)", body)],
        ])
        story.append(img_table)
        story.append(Spacer(1, 16))

    story.append(Paragraph("AI Prediction & Explainability", h2))
    conf_rows = [["Class", "Confidence"]] + [
        [k.capitalize(), f"{v*100:.1f}%"] for k, v in claim["all_confidences"].items()
    ]
    conf_table = Table(conf_rows, colWidths=[150, 150])
    conf_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(Paragraph(
        f"Predicted Severity: <b>{claim['predicted_class'].upper()}</b> "
        f"(confidence {claim['confidence']*100:.1f}%)", body))
    story.append(Spacer(1, 6))
    story.append(conf_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("AI Co-Pilot Summary", h2))
    story.append(Paragraph(claim["summary"], body))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"AI-Suggested Payout: <b>${claim['suggested_payout']:,.2f}</b>", body))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Human-in-the-Loop Decision", h2))
    decision_rows = [
        ["Adjuster Decision", claim["adjuster_decision"].upper()],
        ["Adjuster Notes", claim.get("adjuster_notes") or "-"],
        ["Final Approved Payout", f"${claim['final_payout']:,.2f}"],
    ]
    decision_table = Table(decision_rows, colWidths=[150, 300])
    decision_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(decision_table)

    doc.build(story)
    return out_path
